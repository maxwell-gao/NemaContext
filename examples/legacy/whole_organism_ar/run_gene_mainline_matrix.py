#!/usr/bin/env python3
"""Run the narrowed gene-only mainline matrix."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

project_root = Path(__file__).parent.parent.parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5ad_path", default="dataset/processed/nema_extended_large2025.h5ad")
    p.add_argument("--n_hvg", type=int, default=256)
    p.add_argument("--dt_minutes", type=float, default=40.0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", default=None)
    p.add_argument("--checkpoint_root", default="checkpoints/jit_gene_patch/gene_only_matrix")
    p.add_argument("--run_local_region_baseline", action="store_true", default=True)
    p.add_argument("--skip_local_region_baseline", dest="run_local_region_baseline", action="store_false")
    p.add_argument("--baseline_output_json", default="result/gene_context/local_region_gene_only.json")
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
        "--dt_minutes", str(args.dt_minutes),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--checkpoint_dir", args.checkpoint_root,
        "--patch_composition", "local_only",
        "--no_use_relative_position",
        "--no_use_context_role",
    ]
    if args.device:
        common.extend(["--device", args.device])

    variants = [
        ("gene_only_ctx16_h1", ["--context_size", "16", "--history_patches", "1"]),
        ("gene_only_ctx16_h4", ["--context_size", "16", "--history_patches", "4"]),
        ("gene_only_ctx32_h1", ["--context_size", "32", "--history_patches", "1"]),
        ("gene_only_ctx32_h4", ["--context_size", "32", "--history_patches", "4"]),
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
        run(common + ["--experiment_name", name] + extra, args.dry_run)


if __name__ == "__main__":
    main()
