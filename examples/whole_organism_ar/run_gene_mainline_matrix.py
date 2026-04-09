#!/usr/bin/env python3
"""Run the minimal gene-mainline ablation matrix."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5ad_path", default="dataset/processed/nema_extended_large2025.h5ad")
    p.add_argument("--context_size", type=int, default=32)
    p.add_argument("--n_hvg", type=int, default=256)
    p.add_argument("--dt_minutes", type=float, default=40.0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", default=None)
    p.add_argument("--checkpoint_root", default="checkpoints/jit_gene_patch/mainline_matrix")
    p.add_argument("--run_local_region_baseline", action="store_true", default=True)
    p.add_argument("--skip_local_region_baseline", dest="run_local_region_baseline", action="store_false")
    p.add_argument("--baseline_output_json", default="result/gene_context/local_region_mainline.json")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def run(cmd: list[str], dry_run: bool):
    print(" ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True, cwd=project_root)


def main():
    args = parse_args()
    train_script = project_root / "examples/whole_organism_ar/train_jit_gene_patch.py"
    baseline_script = project_root / "examples/whole_organism_ar/run_local_region_gene_baseline.py"

    common = [
        "uv", "run", "python", str(train_script),
        "--h5ad_path", args.h5ad_path,
        "--n_hvg", str(args.n_hvg),
        "--context_size", str(args.context_size),
        "--dt_minutes", str(args.dt_minutes),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--checkpoint_dir", args.checkpoint_root,
    ]
    if args.device:
        common.extend(["--device", args.device])

    global_context = max(4, (args.context_size - 1) // 4)
    variants = [
        ("no_spatial_local_only_h1", ["--patch_composition", "local_only", "--history_patches", "1", "--no_use_relative_position", "--no_use_context_role"]),
        ("relpos_local_only_h1", ["--patch_composition", "local_only", "--history_patches", "1"]),
        ("relpos_local_only_h4", ["--patch_composition", "local_only", "--history_patches", "4"]),
        ("relpos_local_global_h1", ["--patch_composition", "local_global", "--global_context_size", str(global_context), "--history_patches", "1"]),
    ]

    if args.run_local_region_baseline:
        baseline_cmd = [
            "uv", "run", "python", str(baseline_script),
            "--h5ad_path", args.h5ad_path,
            "--n_hvg", str(args.n_hvg),
            "--dt_minutes", str(args.dt_minutes),
            "--output_json", args.baseline_output_json,
        ]
        if args.device:
            baseline_cmd.extend(["--device", args.device])
        run(baseline_cmd, args.dry_run)

    for name, extra in variants:
        cmd = common + ["--experiment_name", name] + extra
        run(cmd, args.dry_run)


if __name__ == "__main__":
    main()
