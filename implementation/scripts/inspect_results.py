"""Inspect NeuroLens v2 results: encoding heatmap + dissociation matrix.

Prints summary tables and key statistics. Reads from results_v2/ directory.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def inspect_encoding(out_root: Path):
    enc_path = out_root / "encoding_real" / "encoding_aggregate.npz"
    if not enc_path.exists():
        print(f"No encoding aggregate at {enc_path}")
        return
    data = np.load(enc_path, allow_pickle=True)
    mat_raw = data["encoding_matrix_mean"]
    mat_norm = data["encoding_normalized_mean"]
    rois = list(data["roi_names"])
    layers = list(data["layer_indices"])
    nc = data["noise_ceiling_per_roi_mean"]

    print(f"\n=== Encoding model — layer × ROI mean Pearson r ===")
    print(f"matrix shape: {mat_raw.shape}; subjects: {list(data['subjects'])}")

    # Per-ROI: which layer peaks?
    print(f"\nPer-ROI best layer (peak NC-normalized r):")
    print(f"{'ROI':<22} {'best_layer':>10} {'peak_r':>8} {'peak_r/NC':>10} {'NC':>6}")
    for j, roi in enumerate(rois):
        col = mat_norm[:, j]
        if col.sum() == 0:
            print(f"{roi:<22} {'(empty)':>10}")
            continue
        peak_idx = int(np.argmax(col))
        peak_layer = layers[peak_idx]
        peak_r = float(mat_raw[peak_idx, j])
        peak_norm = float(col[peak_idx])
        nc_val = float(nc[j])
        print(f"{roi:<22} {peak_layer:>10d} {peak_r:>8.3f} {peak_norm:>10.3f} {nc_val:>6.3f}")

    # Aggregate by stage (assume layers map to stages from models.yaml)
    print(f"\nPer-layer mean across all ROIs (NC-norm):")
    layer_means = mat_norm.mean(axis=1)
    for l, m in zip(layers, layer_means):
        bar = "█" * int(m * 50)
        print(f"  L{l:2d}: {m:.3f} {bar}")


def inspect_dissociation(out_root: Path):
    dis_path = out_root / "sweep_real" / "dissociation_matrix.npz"
    if not dis_path.exists():
        print(f"\nNo dissociation matrix at {dis_path}")
        return
    d = np.load(dis_path, allow_pickle=True)
    mat = d["matrix"]
    stages = list(d["stage_names"])
    tasks = list(d["task_names"])

    print(f"\n=== Triple Dissociation (stage × task drops) ===")
    header_label = "Patched/Task"
    print(f"{header_label:<15} | " + " | ".join(f"{t:>12}" for t in tasks))
    for i, s in enumerate(stages):
        row = " | ".join(f"{mat[i,j]:>12.4f}" for j in range(len(tasks)))
        print(f"{s:<15} | {row}")

    print(f"\nPeak-drop layer per task:")
    peak_layers = d["peak_layer_per_task"]
    diag_drops = d["diagonal_drops"]
    dissoc_ratio = d["dissociation_ratio"]
    for j, t in enumerate(tasks):
        print(f"  {t:<15}: peak at layer {peak_layers[j]}, "
              f"diag drop {diag_drops[min(j, len(diag_drops)-1)]:.3f}, "
              f"dissoc ratio {dissoc_ratio[min(j, len(dissoc_ratio)-1)]:.2f}")


def inspect_sweep_curves(out_root: Path):
    sweep_root = out_root / "sweep_real"
    print(f"\n=== Per-task layer drop curves ===")
    for task in ("object", "category_mc", "caption"):
        p = sweep_root / f"sweep_{task}.npz"
        if not p.exists():
            print(f"\n{task}: no data")
            continue
        d = np.load(p)
        layers = d["layer_indices"]
        drops = d["drops"]
        baseline = float(d["baseline_score"])
        print(f"\n{task} (baseline={baseline:.3f}):")
        for l, dr in zip(layers, drops):
            bar = "█" * max(0, int(abs(dr) * 50))
            sign = "+" if dr >= 0 else "-"
            print(f"  L{l:2d}: {sign}{abs(dr):.4f} {bar}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    args = parser.parse_args()

    root = Path(args.results_root)
    inspect_encoding(root)
    inspect_dissociation(root)
    inspect_sweep_curves(root)
