"""Heatmap visualizations for NeuroLens.

Layer x ROI alignment heatmaps, multi-model comparisons,
stage boundary visualization, and triple dissociation matrix plots.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for HPC
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

# Publication-quality defaults
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Colorblind-safe palette
CB_COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7',
             '#F0E442', '#56B4E9', '#E69F00', '#000000']


def plot_alignment_heatmap(
    matrix: np.ndarray,
    roi_names: list[str],
    model_name: str,
    metric_name: str = "RSA",
    stage_boundaries: Optional[list[int]] = None,
    noise_ceiling: Optional[dict] = None,
    p_values: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    output_path: Optional[str] = None,
    figsize: tuple = (8, 10),
    cmap: str = "RdBu_r",
) -> plt.Figure:
    """Plot a layer x ROI alignment heatmap.

    Args:
        matrix: (n_layers, n_rois) alignment scores
        roi_names: ROI labels for columns
        model_name: model identifier for title
        metric_name: 'RSA', 'CKA', or 'Encoding'
        stage_boundaries: layer indices separating stages (e.g., [11, 23])
        noise_ceiling: dict with 'upper' and 'lower' bounds per ROI
        p_values: (n_layers, n_rois) p-values for significance
        alpha: significance threshold
        output_path: if provided, save figure
        figsize: figure dimensions
        cmap: colormap

    Returns:
        matplotlib Figure
    """
    n_layers, n_rois = matrix.shape
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    vmax = np.abs(matrix).max()
    vmin = -vmax if matrix.min() < 0 else 0
    im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation='nearest')

    # Significance markers
    if p_values is not None:
        for i in range(n_layers):
            for j in range(n_rois):
                if p_values[i, j] < alpha:
                    ax.text(j, i, '*', ha='center', va='center',
                            fontsize=6, color='black', fontweight='bold')

    # Stage boundary lines
    if stage_boundaries:
        for boundary in stage_boundaries:
            ax.axhline(y=boundary - 0.5, color='white', linewidth=2, linestyle='--')

        # Stage labels
        boundaries = [0] + stage_boundaries + [n_layers]
        stage_names = ["Visual\nEncoding", "Cross-Modal\nFusion", "Linguistic\nRefinement"]
        for idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            if idx < len(stage_names):
                mid = (start + end) / 2
                ax.text(-0.8, mid, stage_names[idx], ha='right', va='center',
                        fontsize=8, fontstyle='italic', color=CB_COLORS[idx])

    # Axis labels
    ax.set_xticks(range(n_rois))
    ax.set_xticklabels(roi_names, rotation=45, ha='right')
    ax.set_yticks(range(0, n_layers, max(1, n_layers // 10)))
    ax.set_yticklabels([f"L{i}" for i in range(0, n_layers, max(1, n_layers // 10))])
    ax.set_xlabel("Brain ROI")
    ax.set_ylabel("MLLM Layer")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(f"{metric_name} Score")

    # Noise ceiling annotation
    if noise_ceiling:
        for j, roi in enumerate(roi_names):
            if roi in noise_ceiling:
                nc = noise_ceiling[roi]
                upper = nc.get("upper", None)
                if upper is not None:
                    ax.axvline(x=j, ymin=0, ymax=0.02, color='gold',
                               linewidth=3, clip_on=False)

    ax.set_title(f"{model_name}: {metric_name} Alignment (Layer x ROI)")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format=Path(output_path).suffix[1:])
        logger.info(f"Saved heatmap to {output_path}")

    return fig


def plot_multi_model_comparison(
    matrices: dict[str, np.ndarray],
    roi_names: list[str],
    metric_name: str = "RSA",
    stage_boundaries_per_model: Optional[dict[str, list[int]]] = None,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot alignment heatmaps for multiple models side by side.

    Args:
        matrices: dict mapping model_name -> (n_layers, n_rois) matrix
        roi_names: shared ROI labels
        metric_name: alignment metric name
        stage_boundaries_per_model: dict mapping model_name -> boundary layers
        output_path: save path
    """
    n_models = len(matrices)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 8))
    if n_models == 1:
        axes = [axes]

    # Find global vmax for consistent colorbar
    global_vmax = max(np.abs(m).max() for m in matrices.values())

    for ax, (model_name, matrix) in zip(axes, matrices.items()):
        n_layers = matrix.shape[0]
        im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r',
                       vmin=-global_vmax, vmax=global_vmax,
                       interpolation='nearest')

        # Stage boundaries
        if stage_boundaries_per_model and model_name in stage_boundaries_per_model:
            for boundary in stage_boundaries_per_model[model_name]:
                ax.axhline(y=boundary - 0.5, color='white', linewidth=2,
                           linestyle='--')

        ax.set_xticks(range(len(roi_names)))
        ax.set_xticklabels(roi_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel("Layer")
        ax.set_title(model_name, fontsize=11)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(f"{metric_name} Score")

    fig.suptitle(f"Cross-Model {metric_name} Alignment", fontsize=13, y=1.02)
    fig.tight_layout(rect=[0, 0, 0.9, 1])

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format=Path(output_path).suffix[1:])
        logger.info(f"Saved multi-model comparison to {output_path}")

    return fig


def plot_dissociation_matrix(
    matrix: np.ndarray,
    stage_labels: list[str] = None,
    task_labels: list[str] = None,
    analogies: Optional[dict] = None,
    output_path: Optional[str] = None,
    figsize: tuple = (8, 7),
) -> plt.Figure:
    """Plot the 3x3 triple dissociation matrix.

    Rows: patched stages, Columns: task types.
    Diagonal should show the largest drops (colored intensely).

    Args:
        matrix: (3, 3) performance drop matrix
        stage_labels: row labels
        task_labels: column labels
        analogies: dict with brain lesion analogy descriptions
        output_path: save path
    """
    if stage_labels is None:
        stage_labels = ["Visual\n(L0-10)", "Fusion\n(L11-22)", "Language\n(L23-31)"]
    if task_labels is None:
        task_labels = ["Visual\nRecognition", "Cross-Modal\nBinding", "Language\nGeneration"]

    fig, ax = plt.subplots(figsize=figsize)

    # Custom colormap: white for no drop, red for large drop
    cmap = sns.color_palette("Reds", as_cmap=True)
    vmax = np.abs(matrix).max()

    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=vmax, aspect='equal')

    # Annotate cells
    for i in range(3):
        for j in range(3):
            value = matrix[i, j]
            # Bold and larger font for diagonal (expected dissociation)
            if i == j:
                ax.text(j, i, f"{value:.3f}", ha='center', va='center',
                        fontsize=14, fontweight='bold', color='white' if value > vmax * 0.5 else 'black')
                # Add a border to highlight diagonal
                rect = mpatches.FancyBboxPatch(
                    (j - 0.48, i - 0.48), 0.96, 0.96,
                    boxstyle="round,pad=0.02",
                    linewidth=3, edgecolor='gold', facecolor='none'
                )
                ax.add_patch(rect)
            else:
                ax.text(j, i, f"{value:.3f}", ha='center', va='center',
                        fontsize=11, color='black')

    ax.set_xticks(range(3))
    ax.set_xticklabels(task_labels, fontsize=10)
    ax.set_yticks(range(3))
    ax.set_yticklabels(stage_labels, fontsize=10)
    ax.set_xlabel("Task Type (measured)", fontsize=11)
    ax.set_ylabel("Patched Stage", fontsize=11)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Performance Drop", fontsize=10)

    ax.set_title("Triple Dissociation: Stage-Specific Causal Effects", fontsize=12)

    # Add legend explaining the expected pattern
    ax.text(0.5, -0.2,
            "Gold border = expected selective impairment (diagonal).\n"
            "Larger value = greater performance drop from patching.",
            transform=ax.transAxes, ha='center', fontsize=8,
            fontstyle='italic', color='gray')

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format=Path(output_path).suffix[1:])
        logger.info(f"Saved dissociation matrix to {output_path}")

    return fig


def plot_layer_profile(
    rsa_per_layer: dict[str, np.ndarray],
    roi_names: list[str],
    model_name: str,
    stage_boundaries: Optional[list[int]] = None,
    noise_ceilings: Optional[dict[str, float]] = None,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot alignment profiles per ROI as line plots across layers.

    One line per ROI, x-axis = layer, y-axis = alignment score.

    Args:
        rsa_per_layer: dict mapping ROI name -> (n_layers,) array of scores
        roi_names: ROI labels
        model_name: model name for title
        stage_boundaries: vertical lines at stage boundaries
        noise_ceilings: dict mapping ROI -> ceiling value
        output_path: save path
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for idx, roi in enumerate(roi_names):
        if roi not in rsa_per_layer:
            continue
        scores = rsa_per_layer[roi]
        color = CB_COLORS[idx % len(CB_COLORS)]
        ax.plot(range(len(scores)), scores, '-o', markersize=3,
                label=roi, color=color, linewidth=1.5)

        # Noise ceiling
        if noise_ceilings and roi in noise_ceilings:
            ax.axhline(y=noise_ceilings[roi], color=color,
                       linestyle=':', linewidth=0.8, alpha=0.5)

    # Stage boundary shading
    if stage_boundaries:
        boundaries = [0] + stage_boundaries + [len(next(iter(rsa_per_layer.values())))]
        colors_bg = ['#E6F3FF', '#FFF3E6', '#E6FFE6']  # light blue, orange, green
        labels_bg = ['Visual', 'Fusion', 'Language']
        for idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            if idx < len(colors_bg):
                ax.axvspan(start - 0.5, end - 0.5, alpha=0.15,
                           color=colors_bg[idx], label=f"_{labels_bg[idx]}")
                ax.text((start + end) / 2, ax.get_ylim()[1] * 0.95,
                        labels_bg[idx], ha='center', fontsize=8, alpha=0.6)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Alignment Score")
    ax.set_title(f"{model_name}: Layer-wise Brain Alignment Profile")
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format=Path(output_path).suffix[1:])
        logger.info(f"Saved layer profile to {output_path}")

    return fig


def plot_metric_comparison(
    rsa_matrix: np.ndarray,
    cka_matrix: np.ndarray,
    encoding_matrix: np.ndarray,
    roi_names: list[str],
    model_name: str,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot comparison of three alignment metrics (RSA, CKA, Encoding).

    Shows whether the metrics converge on the same stage-brain mapping.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    titles = ["RSA (Spearman)", "Linear CKA", "Encoding (Ridge r)"]
    matrices = [rsa_matrix, cka_matrix, encoding_matrix]

    global_vmax = max(np.abs(m).max() for m in matrices if m is not None)

    for ax, title, matrix in zip(axes, titles, matrices):
        if matrix is None:
            ax.text(0.5, 0.5, "Not computed", ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(title)
            continue

        im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r',
                       vmin=-global_vmax, vmax=global_vmax,
                       interpolation='nearest')
        ax.set_xticks(range(len(roi_names)))
        ax.set_xticklabels(roi_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel("Layer")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f"{model_name}: Metric Convergence", fontsize=13)
    fig.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format=Path(output_path).suffix[1:])
        logger.info(f"Saved metric comparison to {output_path}")

    return fig
