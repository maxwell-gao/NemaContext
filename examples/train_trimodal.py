"""Trimodal training: transcriptome + spatial + lineage with masking.

Integrates all three modalities into a single model using modality masking
to handle partial observations. Follows curriculum learning:
    Phase 1: Spatial-only focus
    Phase 2: Transcriptome-only focus
    Phase 3: Joint training

Usage:
    uv run python examples/train_trimodal.py
    uv run python examples/train_trimodal.py --epochs 100 --curriculum phased
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
    DiscreteInterpolatingFlow,
    OUFlow,
    branching_bridge,
)
from src.branching_flows.lineage import batch_lineage_bias
from src.branching_flows.nema_model import NemaFlowModel
from src.branching_flows.trimodal_dataset import TrimodalDataset
from src.branching_flows.trimodal_loss import (
    curriculum_trimodal_loss,
    trimodal_context_loss,
    weak_anchor_loss_masked,
)


def parse_args():
    p = argparse.ArgumentParser(description="Trimodal training with masking")
    p.add_argument(
        "--h5ad_path",
        default="dataset/processed/nema_extended_large2025.h5ad",
        help="Path to extended AnnData file",
    )
    p.add_argument("--n_hvg", type=int, default=2000)
    p.add_argument("--time_bins", type=int, default=10)
    p.add_argument(
        "--ordering", default="random", choices=["random", "lineage", "spatial"]
    )
    p.add_argument(
        "--max_cells_per_bin",
        type=int,
        default=1024,
        help="Max cells per time bin for GPU memory",
    )

    # Model architecture
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--head_dim", type=int, default=32)

    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # Curriculum
    p.add_argument(
        "--curriculum",
        default="phased",
        choices=["none", "phased"],
        help="Curriculum strategy",
    )
    p.add_argument("--phase1_epochs", type=int, default=15)
    p.add_argument("--phase2_epochs", type=int, default=15)

    # Lineage bias
    p.add_argument("--use_lineage_bias", action="store_true", default=True)
    p.add_argument("--lineage_temp", type=float, default=1.0)
    p.add_argument("--no_lineage_bias", action="store_true")

    # Curriculum anchor
    p.add_argument("--weak_anchor_weight", type=float, default=0.1)
    p.add_argument("--anchor_decay_epochs", type=int, default=15)

    # Loss weights
    p.add_argument("--lambda_sinkhorn", type=float, default=1.0)
    p.add_argument("--lambda_count", type=float, default=0.1)
    p.add_argument("--lambda_diversity", type=float, default=0.01)
    p.add_argument("--sinkhorn_blur", type=float, default=0.05)

    # Bridge parameters
    p.add_argument("--deletion_pad", type=float, default=1.2)
    p.add_argument("--branching_time_prob", type=float, default=0.5)

    p.add_argument("--device", default="cpu")
    p.add_argument("--checkpoint_dir", default="checkpoints_trimodal")
    p.add_argument("--log_every", type=int, default=1)
    return p.parse_args()


def count_cells_masked(states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Count valid cells per batch."""
    return mask.sum(dim=-1).float()


def get_training_phase(epoch: int, args) -> int:
    """Determine curriculum phase from epoch."""
    if args.curriculum != "phased":
        return 3  # Joint training

    if epoch <= args.phase1_epochs:
        return 1  # Spatial focus
    elif epoch <= args.phase1_epochs + args.phase2_epochs:
        return 2  # Transcriptome focus
    else:
        return 3  # Joint training


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Handle ablation flag
    if args.no_lineage_bias:
        args.use_lineage_bias = False

    print("Loading trimodal dataset...")
    dataset = TrimodalDataset(
        args.h5ad_path,
        n_hvg=args.n_hvg,
        time_bins=args.time_bins,
        ordering=args.ordering,
        use_lineage_bias=args.use_lineage_bias,
        max_cells_per_bin=args.max_cells_per_bin,
    )

    # Print dataset statistics
    print("Dataset loaded:")
    for key, value in dataset.stats.items():
        print(f"  {key}: {value}")

    print(f"  Time bins: {len(dataset)}")
    print(
        f"  Continuous dim: {dataset.continuous_dim} (genes={dataset._gene_dim}, spatial={dataset._spatial_dim})"
    )
    print(f"  Discrete K: {dataset.K}")

    # Create flow
    flow = CoalescentFlow(
        processes=(
            OUFlow(theta=25.0, var_0=5.0, var_1=0.01),
            DiscreteInterpolatingFlow(K=dataset.K),
        ),
        branch_time_dist=beta(1, 2),
    )

    # Create model
    model = NemaFlowModel(
        continuous_dim=dataset.continuous_dim,
        discrete_K=dataset.K,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        head_dim=args.head_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} params")
    print(
        f"Architecture: d_model={args.d_model}, n_layers={args.n_layers}, n_heads={args.n_heads}"
    )
    print(f"Lineage bias: {args.use_lineage_bias} (temp={args.lineage_temp})")
    print(f"Curriculum: {args.curriculum}")
    print(f"  Phase 1 (spatial): epochs 1-{args.phase1_epochs}")
    print(
        f"  Phase 2 (transcriptome): epochs {args.phase1_epochs + 1}-{args.phase1_epochs + args.phase2_epochs}"
    )
    print(f"  Phase 3 (joint): epochs {args.phase1_epochs + args.phase2_epochs + 1}+")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * max(1, len(dataset) // args.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining for {args.epochs} epochs...")
    indices = list(range(len(dataset)))
    best_loss = float("inf")
    keys = ["total", "sink", "count", "div", "anchor"]

    for epoch in range(1, args.epochs + 1):
        # Determine curriculum phase
        phase = get_training_phase(epoch, args)
        phase_names = {1: "spatial", 2: "transcriptome", 3: "joint"}
        phase_str = phase_names[phase]

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

            # Standard branching bridge
            bo = branching_bridge(
                flow,
                dataset.x0_sampler,
                x1s,
                tb,
                deletion_pad=args.deletion_pad,
                use_branching_time_prob=args.branching_time_prob,
            )
            bo = bo.to(device)

            # Compute lineage bias if enabled
            lineage_bias = None
            if args.use_lineage_bias:
                cell_names_list = [dataset.get_cell_names_at(i) for i in bi]
                actual_len = bo.Xt.states[0].shape[1]
                lineage_bias = batch_lineage_bias(
                    cell_names_list,
                    device=device,
                    temperature=args.lineage_temp,
                    max_seq_len=actual_len,
                )

            # Forward pass
            (xc, xd), hs, hd = model(bo.t, bo.Xt, lineage_bias=lineage_bias)

            mask = bo.Xt.padmask
            act = mask & bo.Xt.flowmask
            B, L = mask.shape

            # Count cells
            pred_count = count_cells_masked(xc, act)
            real_count = count_cells_masked(bo.X1anchor[0], act)

            # Get modality masks from states (need to extract from batch)
            # For now, assume all samples have same mask pattern
            # In practice, you'd collate properly
            modality_masks = torch.ones(B, 3, device=device)  # Default: all available
            if hasattr(x1s[0], "modality_masks"):
                # modality_masks is [N] integer mask per cell, convert to [B, 3] float
                per_sample_masks = []
                for s in x1s:
                    # s.modality_masks is shape [N] with integer values
                    # Convert to binary indicators: has_transcriptome, has_spatial, has_lineage
                    mask_int = s.modality_masks  # [N]
                    # Aggregate: any cell has the modality
                    has_transcriptome = ((mask_int & 1) > 0).any().float()
                    has_spatial = ((mask_int & 2) > 0).any().float()
                    has_lineage = ((mask_int & 4) > 0).any().float()
                    per_sample_masks.append(
                        torch.stack([has_transcriptome, has_spatial, has_lineage])
                    )
                modality_masks = torch.stack(per_sample_masks).to(device)

            # Trimodal loss
            if args.curriculum == "phased":
                loss, loss_dict = curriculum_trimodal_loss(
                    xc,
                    bo.X1anchor[0],
                    pred_count,
                    real_count,
                    modality_masks,
                    gene_dim=dataset._gene_dim,
                    training_phase=phase,
                    lambda_sinkhorn=args.lambda_sinkhorn,
                    lambda_count=args.lambda_count,
                    lambda_diversity=args.lambda_diversity,
                    sinkhorn_blur=args.sinkhorn_blur,
                )
            else:
                loss, loss_dict = trimodal_context_loss(
                    xc,
                    bo.X1anchor[0],
                    pred_count,
                    real_count,
                    modality_masks,
                    gene_dim=dataset._gene_dim,
                    lambda_sinkhorn=args.lambda_sinkhorn,
                    lambda_count=args.lambda_count,
                    lambda_diversity=args.lambda_diversity,
                    sinkhorn_blur=args.sinkhorn_blur,
                )

            # Optional weak anchor
            if anchor_weight > 0:
                anchor_l = weak_anchor_loss_masked(
                    xc,
                    bo.X1anchor[0],
                    act,
                    modality_masks,
                    gene_dim=dataset._gene_dim,
                )
                loss = loss + anchor_weight * anchor_l
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
                f"Epoch {epoch:3d}/{args.epochs} [{phase_str:12s}] | "
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
    print(f"Checkpoints saved to: {ckpt_dir}")


if __name__ == "__main__":
    main()
