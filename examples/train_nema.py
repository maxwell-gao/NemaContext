"""Train a BranchingFlows model on C. elegans data.

End-to-end training script: AnnData → NemaDataset → branching_bridge → model → loss.
Demonstrates the dLLM+BranchingFlows hybrid paradigm on real biological data.

Usage:
    uv run python examples/train_nema.py
    uv run python examples/train_nema.py --epochs 50 --d_model 256 --n_layers 6
    uv run python examples/train_nema.py --ordering lineage --device cuda
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.branching_flows import (
    CoalescentFlow,
    DiscreteInterpolatingFlow,
    OUFlow,
    branching_bridge,
    loss_scale,
    split_loss,
    deletion_loss,
)
from src.branching_flows.nema_dataset import NemaDataset
from src.branching_flows.nema_model import NemaFlowModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BranchingFlows on C. elegans")
    p.add_argument("--data", default="dataset/processed/nema_complete_large2025.h5ad")
    p.add_argument("--n_hvg", type=int, default=2000)
    p.add_argument("--time_bins", type=int, default=10)
    p.add_argument("--ordering", default="random", choices=["random", "lineage", "spatial"])
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--head_dim", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--deletion_pad", type=float, default=1.2)
    p.add_argument("--branching_time_prob", type=float, default=0.5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--log_every", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # --- Data ---
    print(f"Loading data from {args.data} ...")
    dataset = NemaDataset(
        args.data,
        n_hvg=args.n_hvg,
        time_bins=args.time_bins,
        ordering=args.ordering,
    )
    print(f"  {len(dataset)} samples, continuous_dim={dataset.continuous_dim}, K={dataset.K}")
    for i in range(len(dataset)):
        print(f"  Sample {i}: {dataset[i].length} cells")

    # --- Flow ---
    flow = CoalescentFlow(
        processes=(
            OUFlow(theta=10.0, var_0=1.0, var_1=0.001),
            DiscreteInterpolatingFlow(K=dataset.K),
        ),
        branch_time_dist=beta(1, 2),
    )

    # --- Model ---
    model = NemaFlowModel(
        continuous_dim=dataset.continuous_dim,
        discrete_K=dataset.K,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        head_dim=args.head_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters, device={device}")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * max(1, len(dataset) // args.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # --- Checkpoint dir ---
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Training loop ---
    print(f"\nTraining for {args.epochs} epochs ...")
    indices = list(range(len(dataset)))
    global_step = 0
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        epoch_losses = {
            "total": 0.0, "cont": 0.0, "disc": 0.0, "split": 0.0, "del": 0.0,
        }
        epoch_batches = 0
        t_epoch = time.time()

        # Shuffle sample indices
        import random
        random.shuffle(indices)

        # Mini-batches
        for start in range(0, len(indices), args.batch_size):
            batch_idx = indices[start : start + args.batch_size]
            if len(batch_idx) < 2:
                continue

            # --- Construct training targets (CPU) ---
            x1_list = [dataset[i] for i in batch_idx]
            t_batch = torch.rand(len(x1_list))

            bridge_out = branching_bridge(
                flow,
                dataset.x0_sampler,
                x1_list,
                t_batch,
                deletion_pad=args.deletion_pad,
                use_branching_time_prob=args.branching_time_prob,
            )

            # Move to device
            bridge_out = bridge_out.to(device)

            # --- Forward ---
            (x1_cont, x1_disc), hat_splits, hat_del = model(bridge_out.t, bridge_out.Xt)

            # --- Losses ---
            mask = bridge_out.Xt.padmask
            fmask = bridge_out.Xt.flowmask
            active = mask & fmask  # valid + flowable positions

            B, L = mask.shape
            t_scale = loss_scale(bridge_out.t, power=1.0, min_val=0.2)
            t_scale = t_scale.unsqueeze(1).expand(B, L)

            # Base process: continuous MSE
            cont_target = bridge_out.X1anchor[0]
            cont_diff = (x1_cont - cont_target) ** 2
            cont_loss = _masked_mean(cont_diff.sum(dim=-1), active, t_scale)

            # Base process: discrete cross-entropy
            disc_target = bridge_out.X1anchor[1].long()
            disc_logits_flat = x1_disc[active]
            disc_target_flat = disc_target[active]
            if disc_logits_flat.numel() > 0:
                disc_loss = F.cross_entropy(disc_logits_flat, disc_target_flat)
            else:
                disc_loss = torch.tensor(0.0, device=device)

            # Split loss (Bregman Poisson)
            s_loss = split_loss(
                flow.split_transform,
                hat_splits,
                bridge_out.splits_target,
                mask,
                scale=t_scale,
            )

            # Deletion loss (logit BCE)
            d_loss = deletion_loss(
                hat_del,
                bridge_out.del_flags,
                mask,
                scale=t_scale,
            )

            total_loss = cont_loss + disc_loss + s_loss + d_loss

            # --- Backward ---
            optimizer.zero_grad()
            total_loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            # --- Accumulate ---
            epoch_losses["total"] += total_loss.item()
            epoch_losses["cont"] += cont_loss.item()
            epoch_losses["disc"] += disc_loss.item()
            epoch_losses["split"] += s_loss.item()
            epoch_losses["del"] += d_loss.item()
            epoch_batches += 1
            global_step += 1

        # --- Epoch summary ---
        if epoch_batches == 0:
            continue
        for k in epoch_losses:
            epoch_losses[k] /= epoch_batches

        dt = time.time() - t_epoch

        if epoch % args.log_every == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"loss={epoch_losses['total']:.4f} "
                f"(cont={epoch_losses['cont']:.4f} "
                f"disc={epoch_losses['disc']:.4f} "
                f"split={epoch_losses['split']:.4f} "
                f"del={epoch_losses['del']:.4f}) | "
                f"lr={lr_now:.2e} | {dt:.1f}s"
            )

        # Save best
        if epoch_losses["total"] < best_loss:
            best_loss = epoch_losses["total"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "args": vars(args),
            }, ckpt_dir / "best.pt")

    # Save final
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": epoch_losses["total"],
        "args": vars(args),
    }, ckpt_dir / "final.pt")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to {ckpt_dir}/")


def _masked_mean(
    loss: torch.Tensor,
    mask: torch.Tensor,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    mask_f = mask.float()
    if scale is not None:
        loss = loss * scale
    return (loss * mask_f).sum() / mask_f.sum().clamp(min=1.0)


if __name__ == "__main__":
    main()
