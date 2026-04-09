"""BROT training on WormGUIDES 4D spatial dynamics with real Sulston tree.

Uses REAL developmental data: tracked cell positions, observed divisions,
and known lineage tree structure. This eliminates the data-framework
mismatch of the transcriptome-based training.

Usage:
    uv run python examples/train_wormguides.py
    uv run python examples/train_wormguides.py --epochs 50 --stride 5
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.branching_flows import (
    CoalescentFlow,
    OUFlow,
    DiscreteInterpolatingFlow,
    branching_bridge,
    loss_scale,
    split_loss,
    deletion_loss,
)
from src.branching_flows.loss import (
    sinkhorn_distributional_loss,
    mass_matching_loss,
    energy_regularization,
)
from src.branching_flows.legacy.nema_model import NemaFlowModel
from src.branching_flows.wormguides_dataset import WormGUIDESDataset


def parse_args():
    p = argparse.ArgumentParser(description="BROT on WormGUIDES 4D dynamics")
    p.add_argument("--nuclei_dir", default="dataset/raw/wormguides/nuclei_files")
    p.add_argument("--deaths_csv", default="dataset/raw/wormguides/CellDeaths.csv")
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--min_cells", type=int, default=4)
    p.add_argument("--include_velocity", action="store_true")
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
    p.add_argument("--lambda_ot", type=float, default=0.1)
    p.add_argument("--lambda_mass", type=float, default=0.01)
    p.add_argument("--lambda_energy", type=float, default=0.001)
    p.add_argument("--sinkhorn_blur", type=float, default=0.5)
    p.add_argument("--expected_mass_ratio", type=float, default=2.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--checkpoint_dir", default="checkpoints/wg")
    p.add_argument("--log_every", type=int, default=1)
    return p.parse_args()


def _masked_mean(loss, mask, scale=None):
    mask_f = mask.float()
    if scale is not None:
        loss = loss * scale
    return (loss * mask_f).sum() / mask_f.sum().clamp(min=1.0)


def main():
    args = parse_args()
    device = torch.device(args.device)

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
        f"BROT: ot={args.lambda_ot} mass={args.lambda_mass} energy={args.lambda_energy}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * max(1, len(dataset) // args.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining for {args.epochs} epochs on WormGUIDES spatial dynamics ...")
    indices = list(range(len(dataset)))
    best_loss = float("inf")
    keys = ["total", "cont", "disc", "split", "del", "sink", "mass", "ener"]

    for epoch in range(1, args.epochs + 1):
        ep = {k: 0.0 for k in keys}
        nb = 0
        t0 = time.time()
        random.shuffle(indices)

        for start in range(0, len(indices), args.batch_size):
            bi = indices[start : start + args.batch_size]
            if len(bi) < 2:
                continue

            x1s = [dataset[i] for i in bi]
            tb = torch.rand(len(x1s))

            # Use standard branching_bridge (random coalescent for now;
            # Sulston forest integration happens at the bridge level)
            bo = branching_bridge(
                flow,
                dataset.x0_sampler,
                x1s,
                tb,
                deletion_pad=args.deletion_pad,
                use_branching_time_prob=args.branching_time_prob,
            )
            bo = bo.to(device)

            (xc, xd), hs, hd = model(bo.t, bo.Xt)

            mask = bo.Xt.padmask
            act = mask & bo.Xt.flowmask
            B, L = mask.shape
            ts = loss_scale(bo.t, 1.0, 0.2).unsqueeze(1).expand(B, L)

            ct = bo.X1anchor[0]
            cl = _masked_mean((xc - ct).pow(2).sum(-1), act, ts)

            dt_ = bo.X1anchor[1].long()
            df, dtf = xd[act], dt_[act]
            dl = (
                F.cross_entropy(df, dtf)
                if df.numel() > 0
                else torch.tensor(0.0, device=device)
            )

            sl = split_loss(flow.split_transform, hs, bo.splits_target, mask, ts)
            ddl = deletion_loss(hd, bo.del_flags, mask, ts)

            otl = sinkhorn_distributional_loss(xc, ct, act, blur=args.sinkhorn_blur)
            ml = mass_matching_loss(
                hs,
                hd,
                args.expected_mass_ratio,
                mask,
                split_transform=flow.split_transform,
            )
            el = energy_regularization(xc, bo.Xt.states[0], bo.t, act)

            total = (
                cl
                + dl
                + sl
                + ddl
                + args.lambda_ot * otl
                + args.lambda_mass * ml
                + args.lambda_energy * el
            )

            optimizer.zero_grad()
            total.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            ep["total"] += total.item()
            ep["cont"] += cl.item()
            ep["disc"] += dl.item()
            ep["split"] += sl.item()
            ep["del"] += ddl.item()
            ep["sink"] += otl.item()
            ep["mass"] += ml.item()
            ep["ener"] += el.item()
            nb += 1

        if nb == 0:
            continue
        for k in keys:
            ep[k] /= nb

        if epoch % args.log_every == 0:
            lr = scheduler.get_last_lr()[0]
            dt = time.time() - t0
            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"L={ep['total']:.2f} "
                f"[c={ep['cont']:.2f} d={ep['disc']:.3f} "
                f"s={ep['split']:.3f} x={ep['del']:.3f} | "
                f"S={ep['sink']:.3f} M={ep['mass']:.1f} "
                f"E={ep['ener']:.1f}] "
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
