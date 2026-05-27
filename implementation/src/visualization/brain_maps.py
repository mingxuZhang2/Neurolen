"""Brain surface visualizations for NeuroLens.

Creates brain maps showing which regions align with which MLLM stages,
using nilearn for glass brain and surface plots.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


# MNI coordinates for key ROIs (approximate centers)
ROI_MNI_COORDS = {
    "V1": (-8, -80, 4),
    "V2": (-14, -90, 8),
    "V3": (-20, -88, 16),
    "V4": (-28, -76, -10),
    "FFA": (-40, -52, -18),
    "PPA": (-28, -42, -12),
    "LOC": (-44, -76, -4),
    "STS": (-52, -28, 2),
    "AG": (-44, -64, 30),
    "TPJ": (-52, -52, 22),
    "Broca": (-48, 20, 8),
    "Wernicke": (-58, -38, 12),
}

# Stage color mapping
STAGE_COLORS = {
    "visual": "#0072B2",      # blue
    "fusion": "#D55E00",      # orange
    "language": "#009E73",    # green
    "bridge": "#CC79A7",      # pink (for transitional regions)
}


def plot_glass_brain_alignment(
    roi_scores: dict[str, float],
    roi_stage_map: dict[str, str],
    title: str = "NeuroLens: Brain-MLLM Stage Alignment",
    output_path: Optional[str] = None,
    figsize: tuple = (12, 4),
) -> plt.Figure:
    """Plot glass brain with ROIs colored by their best-matching MLLM stage.

    Args:
        roi_scores: dict mapping ROI name -> peak alignment score
        roi_stage_map: dict mapping ROI name -> best matching stage name
        title: figure title
        output_path: save path
    """
    try:
        from nilearn import plotting
    except ImportError:
        logger.warning("nilearn not available; skipping glass brain plot")
        return _plot_schematic_brain(roi_scores, roi_stage_map, title, output_path)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Three views: sagittal, coronal, axial
    for ax, display_mode in zip(axes, ['x', 'y', 'z']):
        display = plotting.plot_glass_brain(
            None, display_mode=display_mode, axes=ax,
            title=None, alpha=0.4
        )

        for roi_name, stage in roi_stage_map.items():
            if roi_name not in ROI_MNI_COORDS:
                continue
            coords = ROI_MNI_COORDS[roi_name]
            score = roi_scores.get(roi_name, 0)
            color = STAGE_COLORS.get(stage, "#888888")
            size = max(20, min(200, score * 500))  # scale by score

            display.add_markers(
                [coords], marker_color=color, marker_size=size,
                alpha=0.7,
            )

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                   markersize=10, label=stage.capitalize())
        for stage, c in STAGE_COLORS.items()
        if stage != "bridge"
    ]
    axes[-1].legend(handles=legend_elements, loc='upper right', fontsize=8)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
        logger.info(f"Saved glass brain to {output_path}")

    return fig


def _plot_schematic_brain(
    roi_scores: dict[str, float],
    roi_stage_map: dict[str, str],
    title: str,
    output_path: Optional[str],
) -> plt.Figure:
    """Fallback: plot a schematic diagram when nilearn is unavailable."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Schematic brain outline (simplified lateral view)
    theta = np.linspace(0, 2 * np.pi, 100)
    # Approximate brain shape as an ellipse
    brain_x = 4 * np.cos(theta) - 0.5
    brain_y = 3 * np.sin(theta) + 0.5
    ax.plot(brain_x, brain_y, 'k-', linewidth=2, alpha=0.3)
    ax.fill(brain_x, brain_y, alpha=0.05, color='gray')

    # Approximate ROI positions on the schematic (normalized to brain shape)
    schematic_coords = {
        "V1": (2.5, -0.5),
        "V2": (2.8, 0.2),
        "V3": (3.0, 0.8),
        "V4": (2.5, 1.2),
        "FFA": (1.5, -1.5),
        "PPA": (1.0, -1.8),
        "LOC": (2.0, -1.0),
        "STS": (-0.5, -0.5),
        "AG": (0.5, 1.8),
        "TPJ": (0.0, 1.5),
        "Broca": (-2.5, 1.0),
        "Wernicke": (-1.5, -0.2),
    }

    for roi_name, (x, y) in schematic_coords.items():
        stage = roi_stage_map.get(roi_name, "visual")
        score = roi_scores.get(roi_name, 0)
        color = STAGE_COLORS.get(stage, "#888888")
        size = max(100, min(1000, score * 3000))

        ax.scatter(x, y, s=size, c=color, alpha=0.6, edgecolors='black',
                   linewidth=1, zorder=5)
        ax.annotate(roi_name, (x, y), ha='center', va='bottom',
                    fontsize=8, fontweight='bold',
                    xytext=(0, 8), textcoords='offset points')

    # Stage legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                   markersize=12, label=f"Stage: {stage.capitalize()}")
        for stage, c in STAGE_COLORS.items()
        if stage != "bridge"
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    ax.set_xlim(-5, 5)
    ax.set_ylim(-3, 4)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12)
    ax.axis('off')

    # Annotations
    ax.text(-4.5, -2.5,
            "Circle size proportional to alignment strength.\n"
            "Color indicates best-matching MLLM stage.",
            fontsize=7, fontstyle='italic', color='gray')

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved schematic brain to {output_path}")

    return fig


def plot_roi_surface(
    nsd_root: str,
    subject: str,
    roi_mask: np.ndarray,
    values: np.ndarray,
    title: str = "",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot brain values on a surface using nilearn.

    Args:
        nsd_root: path to NSD data
        subject: subject ID
        roi_mask: boolean mask for ROI voxels
        values: per-voxel values to plot
        title: figure title
        output_path: save path
    """
    try:
        from nilearn import plotting, datasets, surface
        import nibabel as nib
    except ImportError:
        logger.warning("nilearn not available; skipping surface plot")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                             subplot_kw={'projection': '3d'})

    # Use fsaverage for surface rendering
    fsaverage = datasets.fetch_surf_fsaverage()

    # Create a volume with values at ROI locations
    # This requires knowing the volume shape
    roi_dir = Path(nsd_root) / "nsddata" / "ppdata" / subject / "func1pt8mm" / "roi"
    general_path = roi_dir / "nsdgeneral.nii.gz"

    if general_path.exists():
        ref_img = nib.load(general_path)
        vol_data = np.zeros(ref_img.shape[:3])
        vol_data[roi_mask] = values[:roi_mask.sum()]

        vol_img = nib.Nifti1Image(vol_data, ref_img.affine)

        # Project to surface
        for ax, hemi, surf in zip(axes, ['left', 'right'],
                                   [fsaverage.infl_left, fsaverage.infl_right]):
            surf_data = surface.vol_to_surf(vol_img, surf)
            plotting.plot_surf_stat_map(
                surf, surf_data, hemi=hemi, axes=ax,
                title=f"{hemi.capitalize()}", colorbar=True,
                cmap='RdBu_r', threshold=0.01
            )

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
        logger.info(f"Saved surface plot to {output_path}")

    return fig


def plot_stage_brain_correspondence(
    alignment_results: dict,
    model_name: str,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Create a summary figure showing the stage-brain correspondence.

    Combines a heatmap, profile plot, and brain schematic.

    Args:
        alignment_results: dict with alignment matrices and ROI info
        model_name: model name
        output_path: save path
    """
    fig = plt.figure(figsize=(14, 10))

    # Grid layout: heatmap (left), brain diagram (right top), profile (right bottom)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], hspace=0.3, wspace=0.3)

    ax_heatmap = fig.add_subplot(gs[:, 0])
    ax_brain = fig.add_subplot(gs[0, 1])
    ax_profile = fig.add_subplot(gs[1, 1])

    # Heatmap
    matrix = alignment_results.get("rsa_matrix",
             alignment_results.get("cka_matrix",
             alignment_results.get("encoding_matrix")))
    roi_names = alignment_results.get("roi_names", [])

    if matrix is not None and len(roi_names) > 0:
        im = ax_heatmap.imshow(matrix, aspect='auto', cmap='RdBu_r',
                                interpolation='nearest')
        ax_heatmap.set_xticks(range(len(roi_names)))
        ax_heatmap.set_xticklabels(roi_names, rotation=45, ha='right', fontsize=8)
        ax_heatmap.set_ylabel("Layer")
        ax_heatmap.set_title(f"{model_name}: Alignment Heatmap")
        plt.colorbar(im, ax=ax_heatmap, shrink=0.6)

    # Brain schematic (simplified)
    ax_brain.set_xlim(-5, 5)
    ax_brain.set_ylim(-3, 4)
    ax_brain.set_aspect('equal')
    ax_brain.axis('off')
    ax_brain.set_title("Brain Region Mapping")

    # Draw simplified brain
    theta = np.linspace(0, 2 * np.pi, 100)
    ax_brain.plot(4 * np.cos(theta) - 0.5, 3 * np.sin(theta) + 0.5,
                  'k-', linewidth=1.5, alpha=0.3)

    region_positions = {
        "V1-V4": (2.5, 0, "visual"),
        "STS/AG/TPJ": (0, 0.5, "fusion"),
        "Broca/Wernicke": (-2.5, 0.5, "language"),
    }
    for region, (x, y, stage) in region_positions.items():
        color = STAGE_COLORS[stage]
        ax_brain.scatter(x, y, s=300, c=color, alpha=0.6,
                         edgecolors='black', linewidth=1.5, zorder=5)
        ax_brain.text(x, y - 0.8, region, ha='center', fontsize=8,
                      fontweight='bold')

    # Profile plot (layer-wise alignment for each ROI)
    if matrix is not None:
        for j, roi in enumerate(roi_names):
            color = CB_COLORS[j % len(CB_COLORS)]
            ax_profile.plot(matrix[:, j], label=roi, color=color, linewidth=1.5)
        ax_profile.set_xlabel("Layer")
        ax_profile.set_ylabel("Alignment")
        ax_profile.legend(fontsize=7, loc='best')
        ax_profile.set_title("Layer-wise Alignment Profiles")
        ax_profile.grid(axis='y', alpha=0.3)

    fig.suptitle(f"NeuroLens: {model_name} Stage-Brain Correspondence",
                 fontsize=14, fontweight='bold')

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved correspondence figure to {output_path}")

    return fig


# Convenience constant for colorblind-safe palette
CB_COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7',
             '#F0E442', '#56B4E9', '#E69F00', '#000000']
