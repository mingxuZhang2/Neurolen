"""Representational Similarity Analysis for brain-model alignment.

Compares model RDMs (per layer) to brain RDMs (per ROI) using rsatoolbox.
Includes noise ceiling estimation, bootstrap CIs, and permutation tests.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import rsatoolbox
from scipy import stats
from tqdm import tqdm

logger = logging.getLogger(__name__)


def compare_rdms(
    model_rdm: np.ndarray,
    brain_rdm: np.ndarray,
    method: str = "spearman",
) -> float:
    """Compare two RDM matrices using rank correlation.

    Args:
        model_rdm: (n, n) model dissimilarity matrix
        brain_rdm: (n, n) brain dissimilarity matrix
        method: 'spearman' or 'pearson'

    Returns:
        Correlation coefficient between upper triangles
    """
    # Extract upper triangle (excluding diagonal)
    n = model_rdm.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    model_vec = model_rdm[triu_idx]
    brain_vec = brain_rdm[triu_idx]

    if method == "spearman":
        r, p = stats.spearmanr(model_vec, brain_vec)
    elif method == "pearson":
        r, p = stats.pearsonr(model_vec, brain_vec)
    else:
        raise ValueError(f"Unknown method: {method}")

    return float(r)


def compare_rdms_rsatoolbox(
    model_rdm: rsatoolbox.rdm.RDMs,
    brain_rdm: rsatoolbox.rdm.RDMs,
    method: str = "spearman",
) -> float:
    """Compare RDMs using rsatoolbox's compare function."""
    result = rsatoolbox.rdm.compare(model_rdm, brain_rdm, method=method)
    return float(result.flatten()[0])


def compute_rsa_matrix(
    model_rdms: dict[str, np.ndarray],
    brain_rdms: dict[str, np.ndarray],
    method: str = "spearman",
) -> np.ndarray:
    """Compute full layer x ROI RSA alignment matrix.

    Args:
        model_rdms: dict mapping layer identifiers to (n, n) RDMs
        brain_rdms: dict mapping ROI identifiers to (n, n) RDMs
        method: correlation method

    Returns:
        (n_layers, n_rois) matrix of RSA values
    """
    layer_keys = sorted(model_rdms.keys(),
                        key=lambda x: int(x.split("_")[1]) if "_" in x else 0)
    roi_keys = sorted(brain_rdms.keys())

    n_layers = len(layer_keys)
    n_rois = len(roi_keys)
    rsa_matrix = np.zeros((n_layers, n_rois))

    for i, lk in enumerate(tqdm(layer_keys, desc="RSA layers")):
        for j, rk in enumerate(roi_keys):
            rsa_matrix[i, j] = compare_rdms(model_rdms[lk], brain_rdms[rk], method)

    return rsa_matrix, layer_keys, roi_keys


def bootstrap_rsa(
    model_rdm: np.ndarray,
    brain_rdm: np.ndarray,
    method: str = "spearman",
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """Compute RSA with bootstrap confidence intervals.

    Resamples stimuli (rows and corresponding columns of the RDM).

    Returns:
        dict with keys: 'rsa', 'ci_low', 'ci_high', 'se', 'bootstrap_dist'
    """
    rng = np.random.RandomState(seed)
    n = model_rdm.shape[0]
    observed_rsa = compare_rdms(model_rdm, brain_rdm, method)

    bootstrap_values = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        # Resample stimuli
        idx = rng.choice(n, size=n, replace=True)
        model_sub = model_rdm[np.ix_(idx, idx)]
        brain_sub = brain_rdm[np.ix_(idx, idx)]
        bootstrap_values[b] = compare_rdms(model_sub, brain_sub, method)

    alpha = 1 - confidence_level
    ci_low = np.percentile(bootstrap_values, 100 * alpha / 2)
    ci_high = np.percentile(bootstrap_values, 100 * (1 - alpha / 2))

    return {
        "rsa": observed_rsa,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "se": np.std(bootstrap_values),
        "bootstrap_dist": bootstrap_values,
    }


def permutation_test_rsa(
    model_rdm: np.ndarray,
    brain_rdm: np.ndarray,
    method: str = "spearman",
    n_permutations: int = 5000,
    seed: int = 42,
) -> dict:
    """Test significance of RSA via permutation test.

    Permutes stimulus labels in one RDM and recomputes RSA.

    Returns:
        dict with keys: 'rsa', 'p_value', 'null_distribution'
    """
    rng = np.random.RandomState(seed)
    n = model_rdm.shape[0]
    observed_rsa = compare_rdms(model_rdm, brain_rdm, method)

    null_values = np.zeros(n_permutations)
    for p in range(n_permutations):
        perm = rng.permutation(n)
        brain_perm = brain_rdm[np.ix_(perm, perm)]
        null_values[p] = compare_rdms(model_rdm, brain_perm, method)

    # One-tailed p-value (we expect positive correlation)
    p_value = np.mean(null_values >= observed_rsa)

    return {
        "rsa": observed_rsa,
        "p_value": p_value,
        "null_distribution": null_values,
    }


def noise_ceiling(
    brain_rdms: list[np.ndarray],
    method: str = "spearman",
) -> dict:
    """Compute noise ceiling from multiple subjects' brain RDMs.

    Upper bound: average correlation between each subject's RDM and the
                 group-average RDM (including that subject).
    Lower bound: average correlation between each subject's RDM and the
                 group-average RDM (excluding that subject).

    Args:
        brain_rdms: list of (n, n) brain RDMs, one per subject

    Returns:
        dict with 'upper', 'lower', 'per_subject_upper', 'per_subject_lower'
    """
    n_subjects = len(brain_rdms)
    if n_subjects < 2:
        logger.warning("Need >= 2 subjects for noise ceiling; returning NaN")
        return {"upper": np.nan, "lower": np.nan}

    n = brain_rdms[0].shape[0]
    triu_idx = np.triu_indices(n, k=1)

    # Vectorize all RDMs
    rdm_vecs = np.array([rdm[triu_idx] for rdm in brain_rdms])
    group_mean = rdm_vecs.mean(axis=0)

    upper_vals = []
    lower_vals = []

    for i in range(n_subjects):
        # Upper bound: compare subject i to full group mean (including subject i)
        if method == "spearman":
            r_upper, _ = stats.spearmanr(rdm_vecs[i], group_mean)
        else:
            r_upper, _ = stats.pearsonr(rdm_vecs[i], group_mean)
        upper_vals.append(r_upper)

        # Lower bound: compare subject i to leave-one-out mean
        loo_mean = np.delete(rdm_vecs, i, axis=0).mean(axis=0)
        if method == "spearman":
            r_lower, _ = stats.spearmanr(rdm_vecs[i], loo_mean)
        else:
            r_lower, _ = stats.pearsonr(rdm_vecs[i], loo_mean)
        lower_vals.append(r_lower)

    return {
        "upper": np.mean(upper_vals),
        "lower": np.mean(lower_vals),
        "per_subject_upper": np.array(upper_vals),
        "per_subject_lower": np.array(lower_vals),
    }


def run_full_rsa(
    model_rdm_dir: str,
    brain_rdm_dir: str,
    model_name: str,
    n_layers: int,
    subjects: list[str],
    roi_names: list[str],
    token_type: str = "all",
    method: str = "spearman",
    n_bootstrap: int = 10000,
    n_permutations: int = 5000,
    output_dir: Optional[str] = None,
) -> dict:
    """Run the complete RSA analysis pipeline.

    For each layer x ROI combination:
    1. Load model RDM and brain RDMs
    2. Average brain RDMs across subjects
    3. Compute RSA with bootstrap CI
    4. Compute noise ceiling

    Returns:
        dict with 'rsa_matrix', 'ci_matrix', 'p_values', 'noise_ceilings',
             'layer_keys', 'roi_keys'
    """
    from src.models.model_rdm import load_model_rdm
    from src.data.nsd_loader import load_brain_rdm

    rsa_matrix = np.zeros((n_layers, len(roi_names)))
    ci_low_matrix = np.zeros_like(rsa_matrix)
    ci_high_matrix = np.zeros_like(rsa_matrix)
    p_matrix = np.zeros_like(rsa_matrix)
    noise_ceilings_per_roi = {}

    for j, roi in enumerate(tqdm(roi_names, desc="RSA across ROIs")):
        # Load brain RDMs for all subjects
        subject_brain_rdms = []
        for subj in subjects:
            try:
                brain_rdm = load_brain_rdm(brain_rdm_dir, subj, roi)
                subject_brain_rdms.append(brain_rdm)
            except FileNotFoundError:
                logger.warning(f"Brain RDM missing: {subj}/{roi}")

        if not subject_brain_rdms:
            logger.warning(f"No brain RDMs found for {roi}")
            continue

        # Compute noise ceiling
        nc = noise_ceiling(subject_brain_rdms, method)
        noise_ceilings_per_roi[roi] = nc

        # Average brain RDM across subjects
        avg_brain_rdm = np.mean(subject_brain_rdms, axis=0)

        for i in range(n_layers):
            try:
                model_rdm = load_model_rdm(model_rdm_dir, model_name, i, token_type)
            except FileNotFoundError:
                continue

            # Ensure same dimensions
            n_model = model_rdm.shape[0]
            n_brain = avg_brain_rdm.shape[0]
            n_min = min(n_model, n_brain)
            m_rdm = model_rdm[:n_min, :n_min]
            b_rdm = avg_brain_rdm[:n_min, :n_min]

            # RSA with bootstrap
            result = bootstrap_rsa(m_rdm, b_rdm, method, n_bootstrap)
            rsa_matrix[i, j] = result["rsa"]
            ci_low_matrix[i, j] = result["ci_low"]
            ci_high_matrix[i, j] = result["ci_high"]

            # Permutation test
            perm_result = permutation_test_rsa(m_rdm, b_rdm, method, n_permutations)
            p_matrix[i, j] = perm_result["p_value"]

    results = {
        "rsa_matrix": rsa_matrix,
        "ci_low": ci_low_matrix,
        "ci_high": ci_high_matrix,
        "p_values": p_matrix,
        "noise_ceilings": noise_ceilings_per_roi,
        "roi_names": roi_names,
        "model_name": model_name,
        "token_type": token_type,
        "method": method,
    }

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez(
            out / f"rsa_{model_name}_{token_type}.npz",
            rsa_matrix=rsa_matrix,
            ci_low=ci_low_matrix,
            ci_high=ci_high_matrix,
            p_values=p_matrix,
            roi_names=np.array(roi_names),
        )
        logger.info(f"Saved RSA results to {out}")

    return results


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Run RSA analysis")
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--model-rdm-dir", default="rdms")
    parser.add_argument("--brain-rdm-dir", default="brain_data")
    parser.add_argument("--output-dir", default="results/rsa")
    parser.add_argument("--token-type", default="all")
    parser.add_argument("--method", default="spearman")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--n-permutations", type=int, default=5000)
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

    run_full_rsa(
        model_rdm_dir=args.model_rdm_dir,
        brain_rdm_dir=args.brain_rdm_dir,
        model_name=model_dir_name,
        n_layers=model_cfg["n_layers"],
        subjects=args.subjects,
        roi_names=args.roi_groups,
        token_type=args.token_type,
        method=args.method,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        output_dir=args.output_dir,
    )
