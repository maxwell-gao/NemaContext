"""Emergent Context training on WormGUIDES 4D spatial dynamics.

Uses distribution-level objectives with lineage-aware attention bias.
No per-element supervision—structure emerges from cellular interactions.

Key differences from BROT (train_wormguides.py):
- Lineage bias modulates attention (not training target)
- Sinkhorn divergence as primary loss (no MSE anchor)
- Diversity loss prevents mode collapse
- Optional weak anchor for curriculum learning

Usage:
    uv run python examples/train_emergent.py
    uv run python examples/train_emergent.py --epochs 50 --lineage_temp 1.0
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.branching_flows import (
    CoalescentFlow,
    OUFlow,
    DiscreteInterpolatingFlow,
    branching_bridge,
    batch_lineage_bias,
    weak_anchor_loss,
)
from src.branching_flows.emergent_loss import (
    emergent_context_loss,
)
from src.branching_flows.legacy.nema_model import NemaFlowModel
from src.branching_flows.wormguides_dataset import WormGUIDESDataset


def parse_args():
    p = argparse.ArgumentParser(description="Emergent Context training on WormGUIDES")
    p.add_argument("--nuclei_dir", default="dataset/raw/wormguides/nuclei_files")
    p.add_argument("--deaths_csv", default="dataset/raw/wormguides/CellDeaths.csv")
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--min_cells", type=int, default=4)
    p.add_argument("--include_velocity", action="store_true")

    # Model architecture
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--head_dim", type=int, default=32)

    # Training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # Lineage bias
    p.add_argument("--use_lineage_bias", action="store_true", default=True)
    p.add_argument(
        "--lineage_temp",
        type=float,
        default=1.0,
        help="Temperature for lineage bias (lower = stronger bias)",
    )
    p.add_argument(
        "--no_lineage_bias",
        action="store_true",
        help="Disable lineage bias for ablation",
    )

    # Curriculum learning
    p.add_argument(
        "--weak_anchor_weight",
        type=float,
        default=0.1,
        help="Weight for weak anchor loss (set to 0 to disable)",
    )
    p.add_argument(
        "--anchor_decay_epochs",
        type=int,
        default=10,
        help="Epochs over which to decay anchor weight to 0",
    )

    # Loss weights
    p.add_argument("--lambda_sinkhorn", type=float, default=1.0)
    p.add_argument("--lambda_count", type=float, default=0.1)
    p.add_argument("--lambda_diversity", type=float, default=0.01)
    p.add_argument("--sinkhorn_blur", type=float, default=0.05)

    # Bridge parameters
    p.add_argument("--deletion_pad", type=float, default=1.2)
    p.add_argument("--branching_time_prob", type=float, default=0.5)

    p.add_argument("--device", default="cpu")
    p.add_argument("--checkpoint_dir", default="checkpoints/emergent")
    p.add_argument("--log_every", type=int, default=1)
    return p.parse_args()


def count_cells_masked(states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Count valid (non-padding) cells per batch."""
    return mask.sum(dim=-1).float()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Handle ablation flag
    if args.no_lineage_bias:
        args.use_lineage_bias = False

    print("Loading WormGUIDES 4D data ...")
    dataset = WormGUIDESDataset(
        args.nuclei_dir,
        args.deaths_csv,
        min_cells=args.min_cells,
        stride=args.stride,
        include_velocity=args.include_velocity,
    )
    print(f"  {len(dataset)} samples, dim={dataset.continuous_dim}, K={dataset.K}")
    print(f"  Divisions: {len(dataset.get_division_events())}")
    print(f"  Deaths: {len(dataset.get_death_set())}")
    for i in [0, len(dataset) // 4, len(dataset) // 2, len(dataset) - 1]:
        tp = dataset.timepoints[i]
        s = dataset[i]
        nd = sum(s.del_flags)
        print(f"  Sample {i} (t{tp:03d}): {s.length} cells, {nd} deaths")

    flow = CoalescentFlow(
        processes=(
            OUFlow(theta=25.0, var_0=5.0, var_1=0.01),
            DiscreteInterpolatingFlow(K=dataset.K),
        ),
        branch_time_dist=beta(1, 2),
    )

    model = NemaFlowModel(
        continuous_dim=dataset.continuous_dim,
        discrete_K=dataset.K,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        head_dim=args.head_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")
    print(
        f"Emergent Context: lineage_bias={args.use_lineage_bias} temp={args.lineage_temp}"
    )
    print(
        f"  sinkhorn={args.lambda_sinkhorn} count={args.lambda_count} diversity={args.lambda_diversity}"
    )
    print(
        f"  weak_anchor={args.weak_anchor_weight} decay_epochs={args.anchor_decay_epochs}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * max(1, len(dataset) // args.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining for {args.epochs} epochs (Emergent Context) ...")
    indices = list(range(len(dataset)))
    best_loss = float("inf")
    keys = ["total", "sink", "count", "div", "anchor"]

    for epoch in range(1, args.epochs + 1):
        ep = {k: 0.0 for k in keys}
        nb = 0
        t0 = time.time()
        random.shuffle(indices)

        # Curriculum: decay anchor weight
        if args.anchor_decay_epochs > 0 and epoch <= args.anchor_decay_epochs:
            anchor_weight = args.weak_anchor_weight * (
                1 - (epoch - 1) / args.anchor_decay_epochs
            )
        else:
            anchor_weight = 0.0

        for start in range(0, len(indices), args.batch_size):
            bi = indices[start : start + args.batch_size]
            if len(bi) < 2:
                continue

            x1s = [dataset[i] for i in bi]
            tb = torch.rand(len(x1s))

            # Standard branching bridge (no Sulston supervision)
            bo = branching_bridge(
                flow,
                dataset.x0_sampler,
                x1s,
                tb,
                deletion_pad=args.deletion_pad,
                use_branching_time_prob=args.branching_time_prob,
            )
            bo = bo.to(device)

            # Compute lineage bias if enabled (after bridge to match padded length)
            lineage_bias = None
            if args.use_lineage_bias:
                cell_names_list = [dataset.get_cell_names_at(i) for i in bi]
                # Get actual padded length from bridge output
                actual_len = bo.Xt.states[0].shape[1]
                lineage_bias = batch_lineage_bias(
                    cell_names_list,
                    device=device,
                    temperature=args.lineage_temp,
                    max_seq_len=actual_len,
                )

            # Forward pass with lineage bias
            (xc, xd), hs, hd = model(bo.t, bo.Xt, lineage_bias=lineage_bias)

            mask = bo.Xt.padmask
            act = mask & bo.Xt.flowmask
            B, L = mask.shape

            # Count cells
            pred_count = count_cells_masked(xc, act)
            real_count = count_cells_masked(bo.X1anchor[0], act)

            # Emergent context losses
            loss, loss_dict = emergent_context_loss(
                xc,
                bo.X1anchor[0],
                lineage_bias,
                pred_count,
                real_count,
                lambda_sinkhorn=args.lambda_sinkhorn,
                lambda_count=args.lambda_count,
                lambda_diversity=args.lambda_diversity,
            )

            # Optional weak anchor for curriculum
            if anchor_weight > 0:
                anchor_l = weak_anchor_loss(
                    xc, bo.X1anchor[0], act, weight=anchor_weight
                )
                loss = loss + anchor_l
                loss_dict["anchor"] = anchor_l.item()
            else:
                loss_dict["anchor"] = 0.0

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            ep["total"] += loss_dict["total"]
            ep["sink"] += loss_dict["sinkhorn"]
            ep["count"] += loss_dict["count"]
            ep["div"] += loss_dict["diversity"]
            ep["anchor"] += loss_dict["anchor"]
            nb += 1

        if nb == 0:
            continue
        for k in keys:
            ep[k] /= nb

        if epoch % args.log_every == 0:
            lr = scheduler.get_last_lr()[0]
            dt = time.time() - t0
            anchor_str = f"A={ep['anchor']:.3f}" if anchor_weight > 0 else "A=0"
            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"L={ep['total']:.3f} "
                f"[S={ep['sink']:.3f} C={ep['count']:.2f} D={ep['div']:.4f} {anchor_str}] "
                f"lr={lr:.2e} {dt:.1f}s"
            )

        if ep["total"] < best_loss:
            best_loss = ep["total"]
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "loss": best_loss,
                    "args": vars(args),
                },
                ckpt_dir / "best.pt",
            )

    torch.save(
        {
            "epoch": args.epochs,
            "model": model.state_dict(),
            "loss": ep["total"],
            "args": vars(args),
        },
        ckpt_dir / "final.pt",
    )
    print(f"\nDone. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
