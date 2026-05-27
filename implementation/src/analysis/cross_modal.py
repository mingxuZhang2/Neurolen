"""Cross-modal alignment between MLLM and brain.

Splits the question "where does vision-text interaction happen?" into four maps:
  A. _vis tokens → visual cortex   (same-modality)
  B. _txt tokens → visual cortex   (text-after-vision attention -> visual ROIs)
  C. _vis tokens → language network (vision-driven prediction of language areas)
  D. _txt tokens → language network (same-modality)

Plus a variance-partitioning analysis: for each (layer, ROI) we fit three
encoding models — using vision tokens only, text tokens only, and the
concatenation [vis | txt]. The difference
    interaction = R²_VT - R²_V - R²_T
quantifies *cross-modal synergy* at that layer — information predictive of the
ROI that arises only when vision and text representations are combined.

Outputs:
  - encoding_vis.npz : (n_layers, n_rois) mean Pearson r using _vis activations
  - encoding_txt.npz : (n_layers, n_rois) mean Pearson r using _txt activations
  - variance_partition.npz : per-(layer, ROI) R²_V, R²_T, R²_VT, interaction
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


def _pearson_r_per_voxel(y_pred: np.ndarray, y_true: np.ndarray,
                         eps: float = 1e-10) -> np.ndarray:
    yp = y_pred - y_pred.mean(axis=0, keepdims=True)
    yt = y_true - y_true.mean(axis=0, keepdims=True)
    num = (yp * yt).sum(axis=0)
    den = np.sqrt((yp ** 2).sum(axis=0) * (yt ** 2).sum(axis=0))
    r = num / np.maximum(den, eps)
    return np.where(np.isfinite(r), r, 0.0)


def _fit_and_eval(X: np.ndarray, Y: np.ndarray, n_folds: int,
                  alphas: list[float], n_components: int, seed: int = 42
                  ) -> tuple[np.ndarray, float]:
    """Fit Ridge with K-fold CV, return per-voxel r (out-of-fold) and best alpha."""
    n_stim, n_features = X.shape

    # Standardize features per dim
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
    Y = Y - Y.mean(axis=0, keepdims=True)

    if X.shape[1] > n_components and n_stim > n_components:
        pca = PCA(n_components=n_components, random_state=seed)
        X = pca.fit_transform(X)
    elif X.shape[1] > n_stim:
        pca = PCA(n_components=min(n_stim - 1, X.shape[1]), random_state=seed)
        X = pca.fit_transform(X)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_preds = np.zeros_like(Y)
    best_alphas = []
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        Y_tr, Y_te = Y[train_idx], Y[test_idx]
        best_a, best_score, best_pred = None, -np.inf, None
        for a in alphas:
            ridge = Ridge(alpha=a, fit_intercept=False)
            ridge.fit(X_tr, Y_tr)
            Y_pred = ridge.predict(X_te)
            score = float(np.mean(_pearson_r_per_voxel(Y_pred, Y_te)))
            if score > best_score:
                best_score = score
                best_a = a
                best_pred = Y_pred
        fold_preds[test_idx] = best_pred
        best_alphas.append(best_a)

    r_per_voxel = _pearson_r_per_voxel(fold_preds, Y)
    return r_per_voxel, float(np.median(best_alphas))


@dataclass
class CrossModalResult:
    layer_indices: np.ndarray
    roi_names: list
    roi_stages: list                 # 'visual' / 'fusion' / 'language' per ROI
    r_vis_mean: np.ndarray           # (n_layers, n_rois) mean Pearson r — vis tokens
    r_txt_mean: np.ndarray           # (n_layers, n_rois) mean Pearson r — txt tokens
    r2_V: np.ndarray                 # (n_layers, n_rois) R² from vis
    r2_T: np.ndarray                 # (n_layers, n_rois) R² from txt
    r2_VT: np.ndarray                # (n_layers, n_rois) R² from [vis | txt]
    interaction: np.ndarray          # (n_layers, n_rois) R²_VT - R²_V - R²_T


# Standard NeuroLens ROI groupings for the dissociation summary.
# Used in plotting/aggregation; encoding still runs per-ROI individually.
DEFAULT_VISUAL_ROIS = ["V1", "V2", "V3", "V4", "early_visual",
                       "FFA", "PPA", "EBA", "VWFA", "ventral_stream"]
DEFAULT_FUSION_ROIS = ["STS", "AG", "TPOJ", "lateral_stream", "parietal_stream"]
DEFAULT_LANGUAGE_ROIS = ["Broca", "IFG_extended", "auditory_assoc", "temporal_pole"]


def _stage_for_roi(roi: str) -> str:
    if roi in DEFAULT_VISUAL_ROIS:
        return "visual"
    if roi in DEFAULT_FUSION_ROIS:
        return "fusion"
    if roi in DEFAULT_LANGUAGE_ROIS:
        return "language"
    return "other"


def run_cross_modal_encoding(
    activations_vis: dict[int, np.ndarray],
    activations_txt: dict[int, np.ndarray],
    brain_per_roi: dict[str, dict],   # roi -> {"voxels": (n_stim, n_vox), "ncsnr": ...}
    activation_nsd_ids: np.ndarray,
    n_folds: int = 5,
    alphas: Optional[list[float]] = None,
    n_components: int = 512,
    output_dir: Optional[str | Path] = None,
) -> CrossModalResult:
    """Run the four-quadrant cross-modal encoding analysis.

    For each (layer L, ROI R), fits three Ridge encoding models:
        f_V : X_V(L) -> Y_R
        f_T : X_T(L) -> Y_R
        f_VT: [X_V(L) | X_T(L)] -> Y_R

    Returns mean Pearson r for vis-only and txt-only, and R² for all three
    (vis, txt, joint), so the variance-partitioning interaction term can be
    computed downstream as R²_VT − R²_V − R²_T.
    """
    if alphas is None:
        alphas = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]

    layer_indices = sorted(activations_vis.keys())
    roi_names = list(brain_per_roi.keys())
    roi_stages = [_stage_for_roi(r) for r in roi_names]

    n_layers = len(layer_indices)
    n_rois = len(roi_names)

    r_vis_mean = np.zeros((n_layers, n_rois), dtype=np.float32)
    r_txt_mean = np.zeros((n_layers, n_rois), dtype=np.float32)
    r2_V = np.zeros((n_layers, n_rois), dtype=np.float32)
    r2_T = np.zeros((n_layers, n_rois), dtype=np.float32)
    r2_VT = np.zeros((n_layers, n_rois), dtype=np.float32)

    nsd_id_to_act_idx = {int(nid): i for i, nid in enumerate(activation_nsd_ids)}

    for j, roi in enumerate(roi_names):
        Y_full = brain_per_roi[roi]["voxels"]
        brain_nsd_ids = brain_per_roi[roi]["nsd_ids"]
        if Y_full.shape[1] == 0:
            logger.warning(f"ROI {roi} empty; skipping")
            continue

        # Align activations to brain NSD ID ordering
        keep_brain, keep_act = [], []
        for brow, nid in enumerate(brain_nsd_ids):
            ai = nsd_id_to_act_idx.get(int(nid))
            if ai is not None:
                keep_brain.append(brow)
                keep_act.append(ai)
        if len(keep_brain) < n_folds * 2:
            logger.warning(f"ROI {roi}: only {len(keep_brain)} aligned stim; skip")
            continue
        keep_brain = np.array(keep_brain)
        keep_act = np.array(keep_act)
        Y = Y_full[keep_brain]

        for i, L in enumerate(layer_indices):
            X_V = activations_vis[L][keep_act]
            X_T = activations_txt[L][keep_act]
            X_VT = np.concatenate([X_V, X_T], axis=1)

            r_V, _ = _fit_and_eval(X_V, Y, n_folds, alphas, n_components)
            r_T, _ = _fit_and_eval(X_T, Y, n_folds, alphas, n_components)
            r_VT, _ = _fit_and_eval(X_VT, Y, n_folds, alphas, n_components)

            r_vis_mean[i, j] = float(np.mean(r_V))
            r_txt_mean[i, j] = float(np.mean(r_T))
            r2_V[i, j] = float(np.mean(r_V ** 2))
            r2_T[i, j] = float(np.mean(r_T ** 2))
            r2_VT[i, j] = float(np.mean(r_VT ** 2))

        logger.info(
            f"[{roi:18s}] avg r_vis={r_vis_mean[:, j].mean():.3f}, "
            f"avg r_txt={r_txt_mean[:, j].mean():.3f}, "
            f"avg interaction={(r2_VT[:, j] - r2_V[:, j] - r2_T[:, j]).mean():.4f}"
        )

        # Incremental save per ROI so crashes/timeouts don't lose progress
        if output_dir is not None:
            out_inc = Path(output_dir) / "per_roi"
            out_inc.mkdir(parents=True, exist_ok=True)
            np.savez(
                out_inc / f"{roi}.npz",
                layer_indices=np.array(layer_indices),
                r_vis_mean=r_vis_mean[:, j],
                r_txt_mean=r_txt_mean[:, j],
                r2_V=r2_V[:, j],
                r2_T=r2_T[:, j],
                r2_VT=r2_VT[:, j],
                interaction=(r2_VT[:, j] - r2_V[:, j] - r2_T[:, j]),
            )

    interaction = r2_VT - r2_V - r2_T

    result = CrossModalResult(
        layer_indices=np.array(layer_indices),
        roi_names=roi_names,
        roi_stages=roi_stages,
        r_vis_mean=r_vis_mean,
        r_txt_mean=r_txt_mean,
        r2_V=r2_V,
        r2_T=r2_T,
        r2_VT=r2_VT,
        interaction=interaction,
    )

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez(
            out / "cross_modal.npz",
            layer_indices=result.layer_indices,
            roi_names=np.array(roi_names),
            roi_stages=np.array(roi_stages),
            r_vis_mean=r_vis_mean,
            r_txt_mean=r_txt_mean,
            r2_V=r2_V,
            r2_T=r2_T,
            r2_VT=r2_VT,
            interaction=interaction,
        )
        logger.info(f"Saved cross-modal results to {out / 'cross_modal.npz'}")

    return result
