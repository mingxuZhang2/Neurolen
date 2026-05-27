"""Visualizations for token-level MLLM-brain analyses.

Produces the figure series:
  F1: retinotopy — 24×24 token grid projected to V1 voxel positions
  F2: logit-lens trajectory — top-1 word per token at representative layers
  F3: spatial × semantic transition (punch-line) — per-(token, layer) scatter
  F4: brain hierarchy match — MLLM layer × brain ROI best-fit matrix
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------- F1: retinotopy ----------

def plot_token_retinotopy_on_voxels(
    best_voxel: np.ndarray,           # (n_tokens, n_layers, n_rois)
    layer_idx: int,
    roi_name: str,
    roi_idx: int,
    layer_indices: np.ndarray,
    out_path: str | Path,
    grid_h: int = 24, grid_w: int = 24,
    title_suffix: str = "",
):
    """Show for each token (i, j) in the 24x24 grid, which voxel id it maps to.

    The figure has two panels:
      - Left: token grid colored by best voxel id (rainbow)
      - Right: same colors mapped to a 1D voxel-id axis (sanity)

    If MLLM retains retinotopy, neighboring tokens should map to neighboring
    voxel ids (smooth color gradient), and the panel should "look like" a
    retinotopic map.
    """
    l_pos = int(np.where(layer_indices == layer_idx)[0][0])
    bv = best_voxel[:, l_pos, roi_idx]  # (n_tokens,)
    n_tokens = bv.shape[0]
    assert n_tokens == grid_h * grid_w, \
        f"Token count {n_tokens} ≠ {grid_h}*{grid_w}"

    grid = bv.reshape(grid_h, grid_w)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    im = axes[0].imshow(grid, cmap="hsv", aspect="equal")
    axes[0].set_title(f"Best voxel for each token\n(layer L{layer_idx}, ROI {roi_name})")
    axes[0].set_xlabel("token column (image grid)")
    axes[0].set_ylabel("token row (image grid)")
    fig.colorbar(im, ax=axes[0], label="voxel index")

    # Bar plot: voxel id by token raster index, colored same way
    sorted_idx = np.argsort(bv)
    axes[1].scatter(np.arange(n_tokens), bv, c=bv, cmap="hsv", s=8)
    axes[1].set_xlabel("token raster index (row-major)")
    axes[1].set_ylabel("best voxel id")
    axes[1].set_title("Voxel id by token (sorted by id)")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"F1: Token → voxel retinotopy {title_suffix}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_retinotopy_local_coherence(
    best_voxel: np.ndarray,           # (n_tokens, n_layers, n_rois)
    roi_names: list,
    layer_indices: np.ndarray,
    out_path: str | Path,
    grid_h: int = 24, grid_w: int = 24,
):
    """Quantify how spatial the mapping is at each layer.

    Metric: mean | voxel(tok) - voxel(neighbor) | / std(voxel) over all 4-neighbors.
    Lower = more spatially coherent. Random map → ~1.0; retinotopic → much lower.

    Plotted as layer × ROI heatmap of normalized neighbor-coherence.
    """
    n_tok, n_layers, n_rois = best_voxel.shape
    coh = np.zeros((n_layers, n_rois), dtype=np.float32)

    for L in range(n_layers):
        for r in range(n_rois):
            bv = best_voxel[:, L, r].astype(np.float32).reshape(grid_h, grid_w)
            # 4-neighbor differences
            d_h = np.abs(bv[:, 1:] - bv[:, :-1]).mean()
            d_v = np.abs(bv[1:, :] - bv[:-1, :]).mean()
            std = bv.std() + 1e-8
            coh[L, r] = (d_h + d_v) / 2.0 / std

    fig, ax = plt.subplots(figsize=(max(4, n_rois * 0.7), max(4, n_layers * 0.18)))
    im = ax.imshow(coh, aspect="auto", cmap="viridis_r", origin="lower")
    ax.set_xticks(range(n_rois))
    ax.set_xticklabels(roi_names, rotation=30, ha="right")
    ax.set_yticks(range(0, n_layers, max(1, n_layers // 8)))
    ax.set_yticklabels([str(layer_indices[i]) for i in range(0, n_layers, max(1, n_layers // 8))])
    ax.set_xlabel("ROI")
    ax.set_ylabel("MLLM layer")
    ax.set_title("F1b: Retinotopy coherence (lower = more spatial)")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


# ---------- F2: logit-lens word trajectory ----------

def plot_logit_lens_words(
    logit_ids: np.ndarray,           # (n_img, n_layer, n_tok, K)
    logit_probs: np.ndarray,         # (n_img, n_layer, n_tok, K)
    vocab_inv: dict,                  # int -> str (string keys)
    out_path: str | Path,
    n_example_layers: int = 6,
    n_example_tokens: int = 8,
    image_idx: int = 0,
    grid_h: int = 24, grid_w: int = 24,
):
    """Show top-1 word at each (layer, token) for one example image.

    For each of n_example_layers, plot the 24x24 token grid with the top-1 word
    overlaid at sample positions, showing how token "meaning" evolves with depth.
    """
    n_img, n_layer, n_tok, K = logit_ids.shape
    sel_layers = np.linspace(0, n_layer - 1, n_example_layers, dtype=int)
    # Sample tokens: take a sparse grid
    sample_rows = np.linspace(2, grid_h - 3, n_example_tokens // 2, dtype=int)
    sample_cols = np.linspace(2, grid_w - 3, n_example_tokens // 2, dtype=int)

    fig, axes = plt.subplots(1, n_example_layers, figsize=(3.5 * n_example_layers, 4))
    if n_example_layers == 1:
        axes = [axes]
    for ax, L in zip(axes, sel_layers):
        # Background: top-1 probability heatmap
        top1_prob = logit_probs[image_idx, L, :, 0].reshape(grid_h, grid_w)
        im = ax.imshow(top1_prob, cmap="Greys", aspect="equal", vmin=0,
                        vmax=float(np.max(top1_prob) or 1))
        ax.set_title(f"Layer {L}", fontsize=10)
        for ri in sample_rows:
            for ci in sample_cols:
                tok_idx = ri * grid_w + ci
                tid = int(logit_ids[image_idx, L, tok_idx, 0])
                word = vocab_inv.get(str(tid), f"#{tid}")
                # Strip LLaMA byte-tokenization artefacts (Ġ, ▁, etc.)
                word_clean = word.replace("▁", "").replace("Ġ", "")
                if len(word_clean) > 8:
                    word_clean = word_clean[:7] + "."
                ax.text(ci, ri, word_clean,
                        ha="center", va="center",
                        fontsize=7, color="C3",
                        bbox=dict(boxstyle="round,pad=0.1",
                                  fc="white", ec="none", alpha=0.7))
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(f"F2: Logit-lens word evolution (image {image_idx})", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


# ---------- F3: spatial × semantic transition (punch-line) ----------

def plot_spatial_semantic_transition(
    spatial_r: np.ndarray,    # (n_tokens, n_layers) — max over visual ROIs
    semantic_r: np.ndarray,   # (n_tokens, n_layers) — max over language ROIs
    layer_indices: np.ndarray,
    out_path: str | Path,
    cmap_name: str = "plasma",
):
    """The headline figure — token-layer points on (spatial, semantic) axes.

    Early-layer tokens: high spatial, low semantic (right-bottom)
    Late-layer tokens : low spatial, high semantic (left-top)
    """
    n_tokens, n_layers = spatial_r.shape

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: scatter, color = layer
    cmap = plt.get_cmap(cmap_name)
    for L_pos in range(n_layers):
        c = cmap(L_pos / max(1, n_layers - 1))
        axes[0].scatter(
            spatial_r[:, L_pos], semantic_r[:, L_pos],
            color=c, alpha=0.25, s=8,
            label=f"L{layer_indices[L_pos]}" if L_pos in (0, n_layers // 2, n_layers - 1) else None,
        )
    axes[0].set_xlabel("Spatial encoding r (token → visual cortex)")
    axes[0].set_ylabel("Semantic encoding r (token → language cortex)")
    axes[0].set_title("Per-(token, layer) cloud")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)

    # Right: layer-averaged trajectory
    spat_mean = spatial_r.mean(axis=0)
    sem_mean = semantic_r.mean(axis=0)
    sc = axes[1].scatter(spat_mean, sem_mean, c=layer_indices,
                          cmap=cmap_name, s=80, edgecolors="black")
    for L_pos in range(n_layers):
        axes[1].annotate(str(int(layer_indices[L_pos])),
                          (spat_mean[L_pos], sem_mean[L_pos]),
                          fontsize=7, ha="left", va="bottom")
    axes[1].plot(spat_mean, sem_mean, "-", color="gray", linewidth=0.8, alpha=0.5)
    axes[1].set_xlabel("Spatial encoding r (mean over tokens)")
    axes[1].set_ylabel("Semantic encoding r (mean over tokens)")
    axes[1].set_title("Layer-averaged trajectory")
    axes[1].grid(True, alpha=0.3)
    fig.colorbar(sc, ax=axes[1], label="MLLM layer")

    plt.suptitle(
        "F3: Token vision → language transition\n"
        "Each point = one (token, layer); layer color = depth",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


# ---------- F4: brain hierarchy match ----------

def plot_brain_hierarchy_match(
    spatial_per_roi: np.ndarray,    # (n_tokens, n_layers, n_visual_rois)
    semantic_per_roi: np.ndarray,   # (n_tokens, n_layers, n_language_rois)
    visual_rois: list, language_rois: list,
    layer_indices: np.ndarray,
    out_path: str | Path,
):
    """Layer × ROI heatmap of mean(best_r) across tokens.

    Shows where in cortex each MLLM layer aligns best.
    Expect: early layers align with V1-V4, later layers with Broca/Wernicke.
    """
    # Combined ROI list: visual first, then language
    all_rois = list(visual_rois) + list(language_rois)
    mat = np.concatenate([
        spatial_per_roi.mean(axis=0),       # (n_layers, n_vis)
        semantic_per_roi.mean(axis=0),      # (n_layers, n_lang)
    ], axis=1)

    fig, ax = plt.subplots(figsize=(max(6, len(all_rois) * 0.6),
                                     max(5, len(layer_indices) * 0.18)))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", origin="lower",
                    vmin=0, vmax=max(0.2, float(mat.max())))
    ax.set_xticks(range(len(all_rois)))
    ax.set_xticklabels(all_rois, rotation=45, ha="right")
    ax.set_yticks(range(0, len(layer_indices), max(1, len(layer_indices) // 8)))
    ax.set_yticklabels([
        str(layer_indices[i]) for i in range(0, len(layer_indices),
                                              max(1, len(layer_indices) // 8))
    ])
    ax.set_xlabel("Brain ROI (visual → language)")
    ax.set_ylabel("MLLM layer")
    ax.set_title("F4: Mean per-token encoding r across (layer, ROI)")
    # Vertical line at visual / language boundary
    ax.axvline(len(visual_rois) - 0.5, color="white", linewidth=2.5, alpha=0.7)
    ax.axvline(len(visual_rois) - 0.5, color="black", linewidth=0.8, alpha=0.7)
    fig.colorbar(im, ax=ax, label="mean Pearson r over tokens")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
