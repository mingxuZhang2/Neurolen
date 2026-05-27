"""Token-level brain encoding analyses for MLLM vision tokens.

Two analyses, sharing the same Ridge regression backbone:

  1. **Spatial encoding** (where in cortex does each token live?)
     For each (layer L, vision token i), fit Ridge:
         X_i_L ∈ R^{n_images × pca_dim}   → Y_R ∈ R^{n_images × n_voxels_R}
     where Y_R is the per-voxel response of a low-level visual ROI (V1..V4).
     Per-voxel Pearson r → identify "best voxel" for token i at layer L.
     Output a (n_tokens, n_layers, n_rois) array of (best-voxel coordinates + r).

  2. **Semantic encoding** (what word does each token sound like to the brain?)
     For each (layer L, vision token i):
         a) Take the top-K logit-lens word ids for that token.
         b) Look up an LLM word embedding for each of those K words (LLaVA's own
            input embedding matrix).
         c) Average → 1 semantic vector (4096-d, optionally PCA-compressed).
         d) Ridge regress that vector across stimuli → language-region voxels
            (Broca/Wernicke/AG).
     Output a (n_tokens, n_layers, n_rois) array of mean per-voxel r.

Both feed into the "spatial × semantic" trajectory plot (the punch-line).
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


# ---------- core Ridge utility ----------

def _pearson_r_per_voxel(y_pred: np.ndarray, y_true: np.ndarray,
                          eps: float = 1e-10) -> np.ndarray:
    yp = y_pred - y_pred.mean(axis=0, keepdims=True)
    yt = y_true - y_true.mean(axis=0, keepdims=True)
    num = (yp * yt).sum(axis=0)
    den = np.sqrt((yp ** 2).sum(axis=0) * (yt ** 2).sum(axis=0))
    r = num / np.maximum(den, eps)
    return np.where(np.isfinite(r), r, 0.0)


def _ridge_cv_voxelwise(
    X: np.ndarray, Y: np.ndarray,
    n_folds: int = 5,
    alphas: tuple = (1.0, 10.0, 100.0, 1000.0, 10000.0),
    seed: int = 42,
) -> np.ndarray:
    """Fit Ridge with K-fold CV, return out-of-fold per-voxel Pearson r.

    X: (n_stim, n_features) — assumed already-normalized PCA features
    Y: (n_stim, n_voxels)
    """
    n_stim = X.shape[0]
    Y = Y - Y.mean(axis=0, keepdims=True)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_preds = np.zeros_like(Y)
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        Y_tr, Y_te = Y[train_idx], Y[test_idx]
        # Standardize within fold using train stats
        mu = X_tr.mean(axis=0, keepdims=True)
        sigma = X_tr.std(axis=0, keepdims=True) + 1e-8
        X_tr_s = (X_tr - mu) / sigma
        X_te_s = (X_te - mu) / sigma

        # Pick best alpha by mean per-voxel r on a tiny inner held-out chunk
        best_a, best_score, best_pred = None, -np.inf, None
        n_inner = max(1, len(train_idx) // n_folds)
        X_inner_tr = X_tr_s[:-n_inner]
        X_inner_va = X_tr_s[-n_inner:]
        Y_inner_tr = Y_tr[:-n_inner]
        Y_inner_va = Y_tr[-n_inner:]
        for a in alphas:
            r = Ridge(alpha=a, fit_intercept=False)
            r.fit(X_inner_tr, Y_inner_tr)
            Y_va_pred = r.predict(X_inner_va)
            score = float(np.mean(_pearson_r_per_voxel(Y_va_pred, Y_inner_va)))
            if score > best_score:
                best_score = score
                best_a = a
        # Refit on full fold-train
        ridge = Ridge(alpha=best_a, fit_intercept=False)
        ridge.fit(X_tr_s, Y_tr)
        fold_preds[test_idx] = ridge.predict(X_te_s)

    return _pearson_r_per_voxel(fold_preds, Y)


def _ridge_final_voxelwise(
    X_raw: np.ndarray, Y: np.ndarray,
    n_folds: int = 5,
    pca_dim: int = 256,
    alpha: float = 10000.0,
    seed: int = 42,
) -> np.ndarray:
    """Production-quality Ridge: 5-fold CV with fold-local PCA + standardization.

    X_raw: (n_stim, raw_dim) — raw features (4096-d or pre-PCA'd; PCA applied if raw_dim > pca_dim)
    Y: (n_stim, n_voxels)
    Returns: per-voxel Pearson r from out-of-fold predictions.
    """
    n_stim, raw_dim = X_raw.shape
    Y_centered = Y - Y.mean(axis=0, keepdims=True)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_preds = np.zeros_like(Y_centered)

    for train_idx, test_idx in kf.split(X_raw):
        X_tr_raw = X_raw[train_idx]
        X_te_raw = X_raw[test_idx]
        Y_tr = Y_centered[train_idx]

        # Fold-local PCA (fit on train only)
        if raw_dim > pca_dim and n_stim > pca_dim:
            pca = PCA(n_components=pca_dim, random_state=seed, svd_solver="randomized")
            X_tr = pca.fit_transform(X_tr_raw)
            X_te = pca.transform(X_te_raw)
        else:
            X_tr = X_tr_raw
            X_te = X_te_raw

        # Fold-local standardization (fit on train only)
        mu = X_tr.mean(axis=0, keepdims=True)
        sd = X_tr.std(axis=0, keepdims=True) + 1e-8
        X_tr = (X_tr - mu) / sd
        X_te = (X_te - mu) / sd

        ridge = Ridge(alpha=alpha, fit_intercept=False)
        ridge.fit(X_tr, Y_tr)
        fold_preds[test_idx] = ridge.predict(X_te)

    return _pearson_r_per_voxel(fold_preds, Y_centered)


def _ridge_fast_voxelwise(
    X: np.ndarray, Y: np.ndarray,
    alpha: float = 10000.0,
    test_frac: float = 0.2,
    seed: int = 42,
) -> np.ndarray:
    """Fast single-split Ridge for smoke tests only. NOT for final results."""
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = max(1, int(n * test_frac))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    X_tr, X_te = X[train_idx], X[test_idx]
    Y_tr, Y_te = Y[train_idx], Y[test_idx]
    mu = X_tr.mean(0, keepdims=True)
    sd = X_tr.std(0, keepdims=True) + 1e-8
    X_tr = (X_tr - mu) / sd
    X_te = (X_te - mu) / sd
    Y_tr = Y_tr - Y_tr.mean(0, keepdims=True)

    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(X_tr, Y_tr)
    Y_pred = ridge.predict(X_te)
    return _pearson_r_per_voxel(Y_pred, Y_te)


# ---------- spatial (token → V1..V4 voxel) ----------

@dataclass
class SpatialEncodingResult:
    layer_indices: np.ndarray              # (n_layers,)
    roi_names: list                        # ['V1', 'V2', 'V3', 'V4']
    n_tokens: int                          # 576
    # Per-(token, layer, ROI):
    best_r: np.ndarray                     # (n_tokens, n_layers, n_rois) max r across voxels
    best_voxel: np.ndarray                 # (n_tokens, n_layers, n_rois) argmax voxel idx
    mean_r: np.ndarray                     # (n_tokens, n_layers, n_rois) mean r across voxels


def run_token_spatial_encoding(
    token_activations_per_layer: dict[int, np.ndarray],   # L -> (n_img, n_tok, dim)
    activation_nsd_ids: np.ndarray,
    brain_per_roi: dict[str, dict],                       # roi -> {voxels, nsd_ids, ncsnr}
    n_folds: int = 5,
    pca_dim: int = 256,
    final_mode: bool = False,
    output_dir: Optional[str | Path] = None,
) -> SpatialEncodingResult:
    """For each (token i, layer L, ROI R), fit X_i_L -> Y_R and record best voxel.

    Args:
        final_mode: if True, use _ridge_final_voxelwise with fold-local PCA.
                    if False, use _ridge_fast_voxelwise (smoke test only).
    """
    layers = sorted(token_activations_per_layer.keys())
    roi_names = list(brain_per_roi.keys())
    sample0 = token_activations_per_layer[layers[0]]
    _, n_tokens, feat_dim = sample0.shape

    if final_mode:
        logger.info(f"FINAL MODE: 5-fold CV + fold-local PCA({feat_dim}->{pca_dim})")
    else:
        logger.info(f"FAST MODE: single-split (smoke test only)")

    best_r = np.zeros((n_tokens, len(layers), len(roi_names)), dtype=np.float32)
    best_voxel = np.zeros((n_tokens, len(layers), len(roi_names)), dtype=np.int32)
    mean_r = np.zeros((n_tokens, len(layers), len(roi_names)), dtype=np.float32)

    nsd_to_idx = {int(n): i for i, n in enumerate(activation_nsd_ids)}

    for j, roi in enumerate(roi_names):
        Y_full = brain_per_roi[roi]["voxels"]
        brain_ids = brain_per_roi[roi]["nsd_ids"]
        if Y_full.shape[1] == 0:
            logger.warning(f"ROI {roi}: 0 voxels; skipping")
            continue

        keep_brain, keep_act = [], []
        for brow, nid in enumerate(brain_ids):
            a = nsd_to_idx.get(int(nid))
            if a is not None:
                keep_brain.append(brow)
                keep_act.append(a)
        keep_brain = np.array(keep_brain)
        keep_act = np.array(keep_act)
        if len(keep_brain) < n_folds * 4:
            logger.warning(f"ROI {roi}: only {len(keep_brain)} aligned stim; skip")
            continue
        Y = Y_full[keep_brain].astype(np.float32)

        logger.info(f"ROI {roi}: Y shape={Y.shape}")

        for i_l, L in enumerate(layers):
            X_all = token_activations_per_layer[L]
            X_sel = X_all[keep_act].astype(np.float32)

            # Parallel across tokens using joblib (12 cores → ~10× speedup)
            def _fit_one_token(tok):
                X_tok = X_sel[:, tok, :]
                if final_mode:
                    r_v = _ridge_final_voxelwise(
                        X_tok, Y, n_folds=n_folds, pca_dim=pca_dim)
                else:
                    r_v = _ridge_fast_voxelwise(X_tok, Y)
                return tok, float(r_v.max()), int(np.argmax(r_v)), float(r_v.mean())

            try:
                from joblib import Parallel, delayed
                n_jobs = min(12, n_tokens)
                results_list = Parallel(n_jobs=n_jobs, prefer="threads")(
                    delayed(_fit_one_token)(tok) for tok in range(n_tokens)
                )
            except ImportError:
                results_list = [_fit_one_token(tok) for tok in range(n_tokens)]

            for tok, br, bv, mr in results_list:
                best_r[tok, i_l, j] = br
                best_voxel[tok, i_l, j] = bv
                mean_r[tok, i_l, j] = mr

            logger.info(
                f"  layer {L:>2d}  ROI {roi:<6s}  "
                f"best_r mean={best_r[:, i_l, j].mean():.3f}, "
                f"max={best_r[:, i_l, j].max():.3f}"
            )

            # Incremental save per (ROI, layer) — robust to crashes/timeouts
            if output_dir is not None:
                inc = Path(output_dir) / "per_roi_layer"
                inc.mkdir(parents=True, exist_ok=True)
                np.savez(
                    inc / f"{roi}_L{L}.npz",
                    best_r=best_r[:, i_l, j],
                    best_voxel=best_voxel[:, i_l, j],
                    mean_r=mean_r[:, i_l, j],
                )

    result = SpatialEncodingResult(
        layer_indices=np.array(layers),
        roi_names=roi_names,
        n_tokens=n_tokens,
        best_r=best_r,
        best_voxel=best_voxel,
        mean_r=mean_r,
    )

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez(
            out / "spatial_encoding.npz",
            layer_indices=result.layer_indices,
            roi_names=np.array(roi_names),
            best_r=best_r,
            best_voxel=best_voxel,
            mean_r=mean_r,
        )
        logger.info(f"Saved spatial encoding to {out / 'spatial_encoding.npz'}")

    return result


# ---------- semantic (token → Broca/Wernicke voxel via logit lens) ----------

@dataclass
class SemanticEncodingResult:
    layer_indices: np.ndarray
    roi_names: list
    n_tokens: int
    best_r: np.ndarray
    mean_r: np.ndarray


def build_semantic_vectors(
    logit_ids: np.ndarray,             # (n_img, n_layer, n_tok, K)
    logit_probs: np.ndarray,           # (n_img, n_layer, n_tok, K)
    word_embedding_table: np.ndarray,  # (vocab_size, embed_dim)
) -> np.ndarray:
    """Convert logit-lens top-K word ids into per-token semantic vectors.

    For each (image, layer, token), weighted-average the K word embeddings
    by their probabilities.

    Returns: (n_img, n_layer, n_tok, embed_dim)
    """
    n_img, n_layer, n_tok, K = logit_ids.shape
    embed_dim = word_embedding_table.shape[1]
    out = np.zeros((n_img, n_layer, n_tok, embed_dim), dtype=np.float32)

    # vectorized: ids -> embeddings, weighted sum
    for L in range(n_layer):
        ids_L = logit_ids[:, L]                  # (n_img, n_tok, K)
        probs_L = logit_probs[:, L].astype(np.float32)
        # Normalize probs to sum to 1 across K
        probs_L = probs_L / (probs_L.sum(axis=-1, keepdims=True) + 1e-8)
        # Gather embeddings
        emb = word_embedding_table[ids_L.astype(np.int64)]  # (n_img, n_tok, K, embed_dim)
        out[:, L] = (emb * probs_L[..., None]).sum(axis=-2)
    return out


def _build_semantic_layer(
    logit_ids_L: np.ndarray,           # (n_img, n_tok, K) int32
    logit_probs_L: np.ndarray,         # (n_img, n_tok, K) fp16/fp32
    word_emb_table: np.ndarray,         # (vocab, embed_dim) fp32
) -> np.ndarray:
    """Build one layer's semantic vectors: weighted average of top-K word embeddings.

    Processes per-image to avoid materializing (n_img, n_tok, K, embed_dim) which
    is 94 GB for our shapes. Per-image intermediate is only 94 MB.
    """
    n_img, n_tok, K = logit_ids_L.shape
    embed_dim = word_emb_table.shape[1]
    out = np.zeros((n_img, n_tok, embed_dim), dtype=np.float32)
    for i in range(n_img):
        probs_i = logit_probs_L[i].astype(np.float32)        # (n_tok, K)
        probs_i = probs_i / (probs_i.sum(axis=-1, keepdims=True) + 1e-8)
        emb_i = word_emb_table[logit_ids_L[i].astype(np.int64)]  # (n_tok, K, embed_dim)
        out[i] = (emb_i * probs_i[..., None]).sum(axis=-2)
    return out


def run_token_semantic_encoding(
    logit_ids: np.ndarray,                                # (n_img, n_layer, n_tok, K)
    logit_probs: np.ndarray,                              # (n_img, n_layer, n_tok, K)
    word_emb_table: np.ndarray,                            # (vocab, embed_dim)
    activation_nsd_ids: np.ndarray,
    brain_per_roi: dict[str, dict],                        # language ROIs
    n_folds: int = 5,
    sem_pca_dim: int = 256,
    output_dir: Optional[str | Path] = None,
) -> SemanticEncodingResult:
    """Per-layer streaming Ridge(semantic_vec → voxels).

    Builds semantic vectors *one layer at a time* to avoid the 240GB blow-up
    of materializing (n_img × n_layer × n_tok × embed_dim) all at once.
    """
    n_img, n_layer, n_tokens, K = logit_ids.shape
    embed_dim = word_emb_table.shape[1]
    roi_names = list(brain_per_roi.keys())
    nsd_to_idx = {int(n): i for i, n in enumerate(activation_nsd_ids)}

    best_r = np.zeros((n_tokens, n_layer, len(roi_names)), dtype=np.float32)
    mean_r = np.zeros((n_tokens, n_layer, len(roi_names)), dtype=np.float32)

    # Precompute alignment per ROI once
    roi_align: dict[str, tuple] = {}
    for roi in roi_names:
        Y_full = brain_per_roi[roi]["voxels"]
        brain_ids = brain_per_roi[roi]["nsd_ids"]
        if Y_full.shape[1] == 0:
            continue
        keep_brain, keep_act = [], []
        for brow, nid in enumerate(brain_ids):
            a = nsd_to_idx.get(int(nid))
            if a is not None:
                keep_brain.append(brow)
                keep_act.append(a)
        keep_brain = np.array(keep_brain)
        keep_act = np.array(keep_act)
        if len(keep_brain) < n_folds * 4:
            continue
        Y = Y_full[keep_brain].astype(np.float32)
        roi_align[roi] = (keep_act, Y)

    # Iterate layers (outer loop) so we only build one sem_L at a time
    for L in range(n_layer):
        sem_L = _build_semantic_layer(
            logit_ids[:, L], logit_probs[:, L], word_emb_table
        )   # (n_img, n_tok, embed_dim) — ~24 GB at full precision for 1000×576×4096; fp32

        # Per-layer PCA shared across tokens (fit on subsample for speed)
        flat = sem_L.reshape(-1, embed_dim)
        if sem_pca_dim and embed_dim > sem_pca_dim and flat.shape[0] > sem_pca_dim:
            n_total = flat.shape[0]
            fit_n = min(60000, n_total)
            if n_total > fit_n:
                rng = np.random.default_rng(42 + L)
                fit_idx = rng.choice(n_total, size=fit_n, replace=False)
                fit_data = flat[fit_idx]
            else:
                fit_data = flat
            pca = PCA(n_components=sem_pca_dim, random_state=42,
                      svd_solver="randomized")
            pca.fit(fit_data)
            sem_L_red = pca.transform(flat).reshape(n_img, n_tokens, sem_pca_dim)
            del flat, fit_data
        else:
            sem_L_red = sem_L

        for j, roi in enumerate(roi_names):
            if roi not in roi_align:
                continue
            keep_act, Y = roi_align[roi]
            sem_aligned = sem_L_red[keep_act].astype(np.float32)
            for tok in range(n_tokens):
                X = sem_aligned[:, tok, :]
                r_v = _ridge_fast_voxelwise(X, Y)
                best_r[tok, L, j] = float(r_v.max())
                mean_r[tok, L, j] = float(r_v.mean())

            logger.info(
                f"  layer {L:>2d}  ROI {roi:<10s}  "
                f"best_r mean={best_r[:, L, j].mean():.3f}, "
                f"max={best_r[:, L, j].max():.3f}"
            )

            if output_dir is not None:
                inc = Path(output_dir) / "per_roi_layer"
                inc.mkdir(parents=True, exist_ok=True)
                np.savez(
                    inc / f"{roi}_L{L}.npz",
                    best_r=best_r[:, L, j],
                    mean_r=mean_r[:, L, j],
                )

        # Free per-layer memory before next layer
        del sem_L, sem_L_red

    result = SemanticEncodingResult(
        layer_indices=np.arange(n_layer),
        roi_names=roi_names,
        n_tokens=n_tokens,
        best_r=best_r,
        mean_r=mean_r,
    )

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez(
            out / "semantic_encoding.npz",
            layer_indices=result.layer_indices,
            roi_names=np.array(roi_names),
            best_r=best_r,
            mean_r=mean_r,
        )
        logger.info(f"Saved semantic encoding to {out / 'semantic_encoding.npz'}")

    return result


# ---------- helpers to load token-level outputs ----------

def load_token_activations(activations_dir: str | Path,
                            layers: Optional[list[int]] = None
                            ) -> dict[int, np.ndarray]:
    """Load `layer_{L}_vis_tokens.npy` files into {L: array}."""
    activations_dir = Path(activations_dir)
    if layers is None:
        # Auto-discover
        layers = []
        for p in sorted(activations_dir.glob("layer_*_vis_tokens.npy")):
            stem = p.stem
            # 'layer_{L}_vis_tokens'
            try:
                L = int(stem.split("_")[1])
                layers.append(L)
            except (IndexError, ValueError):
                continue
    out = {}
    for L in layers:
        p = activations_dir / f"layer_{L}_vis_tokens.npy"
        if p.exists():
            out[L] = np.load(p).astype(np.float32)
    return out


def load_logit_lens(activations_dir: str | Path,
                     layers: Optional[list[int]] = None
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Load logit-lens ids and probs into stacked arrays.

    Returns:
      ids:   (n_img, n_layer, n_tok, K)
      probs: (n_img, n_layer, n_tok, K)
    """
    activations_dir = Path(activations_dir)
    if layers is None:
        layers = []
        for p in sorted(activations_dir.glob("layer_*_logit_topk_ids.npy")):
            try:
                L = int(p.stem.split("_")[1])
                layers.append(L)
            except (IndexError, ValueError):
                continue
        layers.sort()
    ids = []
    probs = []
    for L in layers:
        ids.append(np.load(activations_dir / f"layer_{L}_logit_topk_ids.npy"))
        probs.append(np.load(activations_dir / f"layer_{L}_logit_topk_probs.npy"))
    ids = np.stack(ids, axis=1)   # (n_img, n_layer, n_tok, K)
    probs = np.stack(probs, axis=1)
    return ids, probs
