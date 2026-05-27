"""Linear predictivity via ridge regression encoding models.

Predicts brain voxel responses from MLLM layer activations using
ridge regression with cross-validation. Third alignment metric
alongside RSA and CKA.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from tqdm import tqdm

logger = logging.getLogger(__name__)


def linear_predictivity(
    X: np.ndarray,
    Y: np.ndarray,
    n_folds: int = 5,
    alphas: list[float] = None,
    seed: int = 42,
) -> dict:
    """Predict brain voxels Y from model features X using ridge regression.

    Args:
        X: (n_stimuli, n_features) model activations
        Y: (n_stimuli, n_voxels) brain voxel responses
        n_folds: number of CV folds
        alphas: regularization values for RidgeCV
        seed: random seed for KFold

    Returns:
        dict with:
            'mean_r': mean per-voxel Pearson r across folds
            'per_voxel_r': (n_voxels,) array of mean correlations
            'per_fold_r': (n_folds,) array of mean correlations per fold
            'best_alpha': selected regularization strength
    """
    if alphas is None:
        alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    n_voxels = Y.shape[1]

    # Accumulate per-voxel correlations across folds
    voxel_r_sum = np.zeros(n_voxels)
    voxel_r_count = np.zeros(n_voxels)
    fold_mean_r = []
    best_alphas = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        # Fit ridge regression with nested CV for alpha selection
        ridge = RidgeCV(alphas=alphas, store_cv_results=False)
        ridge.fit(X_train, Y_train)
        best_alphas.append(ridge.alpha_)

        Y_pred = ridge.predict(X_test)

        # Per-voxel Pearson correlation
        fold_rs = np.zeros(n_voxels)
        for v in range(n_voxels):
            if np.std(Y_test[:, v]) < 1e-10 or np.std(Y_pred[:, v]) < 1e-10:
                fold_rs[v] = 0.0
            else:
                r, _ = stats.pearsonr(Y_pred[:, v], Y_test[:, v])
                fold_rs[v] = r if np.isfinite(r) else 0.0

        voxel_r_sum += fold_rs
        voxel_r_count += 1
        fold_mean_r.append(np.mean(fold_rs))

    per_voxel_r = voxel_r_sum / np.maximum(voxel_r_count, 1)

    return {
        "mean_r": float(np.mean(per_voxel_r)),
        "median_r": float(np.median(per_voxel_r)),
        "per_voxel_r": per_voxel_r,
        "per_fold_r": np.array(fold_mean_r),
        "best_alpha": float(np.median(best_alphas)),
        "n_voxels": n_voxels,
    }


def compute_encoding_matrix(
    activations_dir: str,
    brain_data_dir: str,
    model_name: str,
    n_layers: int,
    subjects: list[str],
    roi_names: list[str],
    token_type: str = "all",
    n_folds: int = 5,
    alphas: list[float] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """Compute full layer x ROI encoding model matrix.

    For each layer x ROI, trains ridge regression and evaluates
    cross-validated Pearson r.

    Returns:
        dict with encoding matrix and per-voxel details
    """
    from src.data.nsd_loader import load_voxel_data

    act_dir = Path(activations_dir) / model_name
    enc_matrix = np.zeros((n_layers, len(roi_names)))
    enc_median = np.zeros_like(enc_matrix)

    for j, roi in enumerate(tqdm(roi_names, desc="Encoding models")):
        # Average voxel data across subjects
        all_voxels = []
        for subj in subjects:
            try:
                vox = load_voxel_data(brain_data_dir, subj, roi)
                all_voxels.append(vox)
            except FileNotFoundError:
                pass

        if not all_voxels:
            logger.warning(f"No voxel data for ROI: {roi}")
            continue

        n_stimuli = min(v.shape[0] for v in all_voxels)
        avg_voxels = np.mean([v[:n_stimuli] for v in all_voxels], axis=0)

        for i in range(n_layers):
            act_path = act_dir / f"layer_{i}_{token_type}.npy"
            if not act_path.exists():
                continue
            activations = np.load(act_path)

            n_min = min(activations.shape[0], avg_voxels.shape[0])
            X = activations[:n_min]
            Y = avg_voxels[:n_min]

            if X.shape[0] < n_folds * 2:
                logger.warning(f"Too few samples for CV: {X.shape[0]}")
                continue

            result = linear_predictivity(X, Y, n_folds, alphas)
            enc_matrix[i, j] = result["mean_r"]
            enc_median[i, j] = result["median_r"]
            logger.debug(f"Layer {i}, ROI {roi}: mean_r={result['mean_r']:.4f}, "
                         f"alpha={result['best_alpha']}")

    results = {
        "encoding_matrix": enc_matrix,
        "encoding_median": enc_median,
        "roi_names": roi_names,
        "model_name": model_name,
        "token_type": token_type,
    }

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez(
            out / f"encoding_{model_name}_{token_type}.npz",
            encoding_matrix=enc_matrix,
            encoding_median=enc_median,
            roi_names=np.array(roi_names),
        )
        logger.info(f"Saved encoding results to {out}")

    return results


def per_subject_encoding(
    activations_dir: str,
    brain_data_dir: str,
    model_name: str,
    layer: int,
    subjects: list[str],
    roi: str,
    token_type: str = "all",
    n_folds: int = 5,
    alphas: list[float] = None,
) -> dict:
    """Run encoding model for a specific layer/ROI per subject.

    Useful for noise ceiling computation and individual variability analysis.
    """
    from src.data.nsd_loader import load_voxel_data

    act_path = Path(activations_dir) / model_name / f"layer_{layer}_{token_type}.npy"
    activations = np.load(act_path)

    per_subj_results = {}
    for subj in subjects:
        try:
            voxels = load_voxel_data(brain_data_dir, subj, roi)
        except FileNotFoundError:
            continue

        n_min = min(activations.shape[0], voxels.shape[0])
        result = linear_predictivity(
            activations[:n_min], voxels[:n_min], n_folds, alphas
        )
        per_subj_results[subj] = result

    return per_subj_results


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Run encoding model analysis")
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--activations-dir", default="activations")
    parser.add_argument("--brain-data-dir", default="brain_data")
    parser.add_argument("--output-dir", default="results/encoding")
    parser.add_argument("--token-type", default="all")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--subjects", nargs="+",
                        default=[f"subj{i:02d}" for i in range(1, 9)])
    parser.add_argument("--roi-groups", nargs="+",
                        default=["visual_early", "multimodal_integration",
                                 "language_network"])

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    with open(args.config) as f:
        config = yaml.safe_load(f)
    model_cfg = config["models"][args.model]
    model_dir_name = model_cfg["name"].replace(" ", "_").replace("-", "_")

    compute_encoding_matrix(
        activations_dir=args.activations_dir,
        brain_data_dir=args.brain_data_dir,
        model_name=model_dir_name,
        n_layers=model_cfg["n_layers"],
        subjects=args.subjects,
        roi_names=args.roi_groups,
        token_type=args.token_type,
        n_folds=args.n_folds,
        output_dir=args.output_dir,
    )
