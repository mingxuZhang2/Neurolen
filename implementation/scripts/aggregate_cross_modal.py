"""Aggregate per-ROI cross_modal results into the main cross_modal.npz and plots."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.analysis.cross_modal import _stage_for_roi

results_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results_v2")
per_roi_dir = results_root / "cross_modal_real" / "per_roi"
out_dir = results_root / "cross_modal_real"
fig_dir = results_root / "figures_real"
fig_dir.mkdir(parents=True, exist_ok=True)

# Canonical ROI ordering — visual → fusion → language
roi_order = [
    "V1", "V2", "V3", "V4", "early_visual",
    "FFA", "PPA", "EBA", "VWFA", "ventral_stream",
    "STS", "AG", "TPOJ", "lateral_stream", "parietal_stream",
    "Broca", "IFG_extended", "auditory_assoc", "temporal_pole",
]
available = []
for roi in roi_order:
    p = per_roi_dir / f"{roi}.npz"
    if p.exists():
        available.append(roi)
print(f"Aggregating {len(available)} ROIs")

# Load all
sample = np.load(per_roi_dir / f"{available[0]}.npz")
layers = sample["layer_indices"]
n_layers = len(layers)
n_rois = len(available)

r_vis = np.zeros((n_layers, n_rois), dtype=np.float32)
r_txt = np.zeros((n_layers, n_rois), dtype=np.float32)
r2_V = np.zeros((n_layers, n_rois), dtype=np.float32)
r2_T = np.zeros((n_layers, n_rois), dtype=np.float32)
r2_VT = np.zeros((n_layers, n_rois), dtype=np.float32)
interaction = np.zeros((n_layers, n_rois), dtype=np.float32)
stages = []

for j, roi in enumerate(available):
    d = np.load(per_roi_dir / f"{roi}.npz")
    r_vis[:, j] = d["r_vis_mean"]
    r_txt[:, j] = d["r_txt_mean"]
    r2_V[:, j] = d["r2_V"]
    r2_T[:, j] = d["r2_T"]
    r2_VT[:, j] = d["r2_VT"]
    interaction[:, j] = d["interaction"]
    stages.append(_stage_for_roi(roi))

np.savez(out_dir / "cross_modal.npz",
         layer_indices=layers,
         roi_names=np.array(available),
         roi_stages=np.array(stages),
         r_vis_mean=r_vis,
         r_txt_mean=r_txt,
         r2_V=r2_V, r2_T=r2_T, r2_VT=r2_VT,
         interaction=interaction)
print(f"Saved {out_dir}/cross_modal.npz")

# Plot 1: 2-panel heatmap (vis | txt)
def heatmap(ax, mat, title, cmap, vmin, vmax):
    im = ax.imshow(mat, aspect="auto", cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(available)))
    ax.set_xticklabels(available, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(0, len(layers), 4))
    ax.set_yticklabels(layers[::4])
    ax.set_xlabel("Brain ROI")
    ax.set_ylabel("LLaVA layer")
    ax.set_title(title)
    # Stage boundary lines
    boundaries = []
    for i in range(len(stages) - 1):
        if stages[i] != stages[i + 1]:
            boundaries.append(i + 0.5)
    for b in boundaries:
        ax.axvline(b, color="white", linewidth=2.5, alpha=0.7)
        ax.axvline(b, color="black", linewidth=0.8, alpha=0.7)
    return im

vmax = float(max(np.abs(r_vis).max(), np.abs(r_txt).max(), 0.05))
fig, axes = plt.subplots(1, 2, figsize=(20, 7))
im1 = heatmap(axes[0], r_vis, "Vision tokens → ROI (Pearson r)", "viridis", 0, vmax)
im2 = heatmap(axes[1], r_txt, "Text tokens → ROI (Pearson r)", "viridis", 0, vmax)
fig.colorbar(im1, ax=axes[0])
fig.colorbar(im2, ax=axes[1])
plt.suptitle("Cross-modal encoding: which modality drives prediction?", fontsize=14)
plt.tight_layout()
plt.savefig(fig_dir / "cm_vis_txt_heatmaps.png", dpi=180)
plt.close(fig)

# Plot 2: difference map (vis - txt)
diff = r_vis - r_txt
abs_max = float(max(np.abs(diff).max(), 0.01))
fig, ax = plt.subplots(figsize=(11, 7))
im = heatmap(ax, diff, "Vision − Text token predictability (Pearson r)", "RdBu_r",
             -abs_max, abs_max)
fig.colorbar(im, ax=ax, label="r_vis − r_txt")
plt.tight_layout()
plt.savefig(fig_dir / "cm_vis_minus_txt.png", dpi=180)
plt.close(fig)

# Plot 3: interaction map
inter_max = float(max(np.abs(interaction).max(), 1e-3))
fig, ax = plt.subplots(figsize=(11, 7))
im = heatmap(ax, interaction, "Variance-partitioning interaction (R²_VT − R²_V − R²_T)",
             "RdBu_r", -inter_max, inter_max)
fig.colorbar(im, ax=ax, label="interaction R²")
plt.tight_layout()
plt.savefig(fig_dir / "cm_interaction.png", dpi=180)
plt.close(fig)

# Plot 4: cross-modal flow trajectories (averaged within stage)
vis_idx = [i for i, s in enumerate(stages) if s == "visual"]
fus_idx = [i for i, s in enumerate(stages) if s == "fusion"]
lang_idx = [i for i, s in enumerate(stages) if s == "language"]

fig, ax = plt.subplots(figsize=(12, 5))
if vis_idx:
    ax.plot(layers, r_vis[:, vis_idx].mean(axis=1), "-o", color="C0",
            label="vis → visual ROIs", linewidth=2)
    ax.plot(layers, r_txt[:, vis_idx].mean(axis=1), "--s", color="C0",
            label="txt → visual ROIs", linewidth=2, alpha=0.7)
if fus_idx:
    ax.plot(layers, r_vis[:, fus_idx].mean(axis=1), "-o", color="C2",
            label="vis → fusion ROIs", linewidth=2)
    ax.plot(layers, r_txt[:, fus_idx].mean(axis=1), "--s", color="C2",
            label="txt → fusion ROIs", linewidth=2, alpha=0.7)
if lang_idx:
    ax.plot(layers, r_vis[:, lang_idx].mean(axis=1), "-o", color="C3",
            label="vis → language ROIs", linewidth=2)
    ax.plot(layers, r_txt[:, lang_idx].mean(axis=1), "--s", color="C3",
            label="txt → language ROIs", linewidth=2, alpha=0.7)
ax.set_xlabel("LLaVA layer")
ax.set_ylabel("Mean Pearson r across stage's ROIs")
ax.set_title("Layer-resolved cross-modal flow: where does vision-text interaction happen?")
ax.legend(loc="best", ncol=2, fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir / "cm_flow.png", dpi=180)
plt.close(fig)

# Summary table
print("\n=== Summary ===")
print(f"{'ROI':<20} {'stage':<10} {'r_vis (avg)':>12} {'r_txt (avg)':>12} {'diff':>8} {'interact':>10}")
for j, roi in enumerate(available):
    print(f"{roi:<20} {stages[j]:<10} "
          f"{r_vis[:, j].mean():>12.4f} {r_txt[:, j].mean():>12.4f} "
          f"{(r_vis[:, j] - r_txt[:, j]).mean():>+8.4f} "
          f"{interaction[:, j].mean():>+10.4f}")
