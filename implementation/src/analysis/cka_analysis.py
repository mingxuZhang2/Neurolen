"""Centered Kernel Alignment (CKA) for brain-model alignment.

Computes linear CKA between MLLM layer activations and brain voxel patterns.
Supports kernel CKA as supplementary analysis.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats
from tqdm import tqdm

logger = logging.getLogger(__name__)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA between two representation matrices.

    CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

    Args:
        X: (n, p) matrix, e.g. MLLM layer activations
        Y: (n, q) matrix, e.g. brain voxel patterns

    Returns:
        CKA score in [0, 1]
    """
    # Center both matrices
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # HSIC terms
    hsic_xy = np.linalg.norm(Y.T @ X, 'fro') ** 2
    hsic_xx = np.linalg.norm(X.T @ X, 'fro')
    hsic_yy = np.linalg.norm(Y.T @ Y, 'fro')

    denom = hsic_xx * hsic_yy
    if denom < 1e-10:
        return 0.0
    return float(hsic_xy / denom)


def kernel_cka(X: np.ndarray, Y: np.ndarray, sigma: Optional[float] = None) -> float:
    """Compute kernel CKA using RBF kernels.

    Args:
        X: (n, p) matrix
        Y: (n, q) matrix
        sigma: RBF kernel bandwidth. If None, uses median heuristic.

    Returns:
        Kernel CKA score
    """
    def rbf_kernel(Z, sigma):
        sq_dists = np.sum(Z ** 2, axis=1, keepdims=True) + \
                   np.sum(Z ** 2, axis=1, keepdims=False) - 2 * Z @ Z.T
        return np.exp(-sq_dists / (2 * sigma ** 2))

    def center_kernel(K):
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    if sigma is None:
        # Median heuristic for bandwidth
        from scipy.spatial.distance import pdist
        sigma_x = np.median(pdist(X, 'euclidean'))
        sigma_y = np.median(pdist(Y, 'euclidean'))
        sigma = (sigma_x + sigma_y) / 2
        if sigma < 1e-10:
            sigma = 1.0

    K_X = center_kernel(rbf_kernel(X, sigma))
    K_Y = center_kernel(rbf_kernel(Y, sigma))

    hsic_xy = np.trace(K_X @ K_Y) / ((X.shape[0] - 1) ** 2)
    hsic_xx = np.trace(K_X @ K_X) / ((X.shape[0] - 1) ** 2)
    hsic_yy = np.trace(K_Y @ K_Y) / ((X.shape[0] - 1) ** 2)

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 0.0
    return float(hsic_xy / denom)


def bootstrap_cka(
    X: np.ndarray,
    Y: np.ndarray,
    method: str = "linear",
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
    sigma: Optional[float] = None,
) -> dict:
    """Compute CKA with bootstrap confidence intervals.

    Args:
        X: (n, p) model activations
        Y: (n, q) brain voxels
        method: 'linear' or 'kernel'
        n_bootstrap: number of bootstrap samples
        confidence_level: CI level
        seed: random seed
        sigma: RBF bandwidth for kernel CKA

    Returns:
        dict with 'cka', 'ci_low', 'ci_high', 'se'
    """
    rng = np.random.RandomState(seed)
    cka_fn = linear_cka if method == "linear" else lambda x, y: kernel_cka(x, y, sigma)

    observed = cka_fn(X, Y)
    n = X.shape[0]
    boot_values = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_values[b] = cka_fn(X[idx], Y[idx])

    alpha = 1 - confidence_level
    ci_low = np.percentile(boot_values, 100 * alpha / 2)
    ci_high = np.percentile(boot_values, 100 * (1 - alpha / 2))

    return {
        "cka": observed,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "se": np.std(boot_values),
    }


def permutation_test_cka(
    X: np.ndarray,
    Y: np.ndarray,
    method: str = "linear",
    n_permutations: int = 5000,
    seed: int = 42,
) -> dict:
    """Test significance of CKA via permutation test."""
    rng = np.random.RandomState(seed)
    cka_fn = linear_cka if method == "linear" else kernel_cka

    observed = cka_fn(X, Y)
    null_values = np.zeros(n_permutations)

    for p in range(n_permutations):
        perm = rng.permutation(X.shape[0])
        null_values[p] = cka_fn(X[perm], Y)

    p_value = np.mean(null_values >= observed)
    return {
        "cka": observed,
        "p_value": p_value,
        "null_distribution": null_values,
    }


def compute_cka_matrix(
    activations_dir: str,
    brain_data_dir: str,
    model_name: str,
    n_layers: int,
    subjects: list[str],
    roi_names: list[str],
    token_type: str = "all",
    method: str = "linear",
    n_bootstrap: int = 10000,
    output_dir: Optional[str] = None,
) -> dict:
    """Compute full layer x ROI CKA alignment matrix.

    Args:
        activations_dir: directory with model activation .npy files
        brain_data_dir: directory with brain voxel .npy files
        model_name: model directory name
        n_layers: number of decoder layers
        subjects: list of subject IDs
        roi_names: list of ROI group names
        token_type: which token type to use
        method: 'linear' or 'kernel'
        n_bootstrap: bootstrap iterations
        output_dir: where to save results

    Returns:
        dict with CKA matrix and statistics
    """
    from src.data.nsd_loader import load_voxel_data

    act_dir = Path(activations_dir) / model_name
    cka_matrix = np.zeros((n_layers, len(roi_names)))
    ci_low_matrix = np.zeros_like(cka_matrix)
    ci_high_matrix = np.zeros_like(cka_matrix)

    for j, roi in enumerate(tqdm(roi_names, desc="CKA across ROIs")):
        # Average brain voxel data across subjects
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

        # Average across subjects (align on stimulus dimension)
        n_stimuli = min(v.shape[0] for v in all_voxels)
        avg_voxels = np.mean([v[:n_stimuli] for v in all_voxels], axis=0)

        for i in range(n_layers):
            act_path = act_dir / f"layer_{i}_{token_type}.npy"
            if not act_path.exists():
                continue
            activations = np.load(act_path)

            # Align stimulus counts
            n_min = min(activations.shape[0], avg_voxels.shape[0])
            X = activations[:n_min]
            Y = avg_voxels[:n_min]

            result = bootstrap_cka(X, Y, method, n_bootstrap)
            cka_matrix[i, j] = result["cka"]
            ci_low_matrix[i, j] = result["ci_low"]
            ci_high_matrix[i, j] = result["ci_high"]

    results = {
        "cka_matrix": cka_matrix,
        "ci_low": ci_low_matrix,
        "ci_high": ci_high_matrix,
        "roi_names": roi_names,
        "model_name": model_name,
        "token_type": token_type,
        "method": method,
    }

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez(
            out / f"cka_{model_name}_{token_type}_{method}.npz",
            cka_matrix=cka_matrix,
            ci_low=ci_low_matrix,
            ci_high=ci_high_matrix,
            roi_names=np.array(roi_names),
        )
        logger.info(f"Saved CKA results to {out}")

    return results


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Run CKA analysis")
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--activations-dir", default="activations")
    parser.add_argument("--brain-data-dir", default="brain_data")
    parser.add_argument("--output-dir", default="results/cka")
    parser.add_argument("--token-type", default="all")
    parser.add_argument("--method", default="linear", choices=["linear", "kernel"])
    parser.add_argument("--n-bootstrap", type=int, default=10000)
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

    compute_cka_matrix(
        activations_dir=args.activations_dir,
        brain_data_dir=args.brain_data_dir,
        model_name=model_dir_name,
        n_layers=model_cfg["n_layers"],
        subjects=args.subjects,
        roi_names=args.roi_groups,
        token_type=args.token_type,
        method=args.method,
        n_bootstrap=args.n_bootstrap,
        output_dir=args.output_dir,
    )
