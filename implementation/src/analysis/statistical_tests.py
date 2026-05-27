"""Statistical testing utilities for NeuroLens.

Bootstrap confidence intervals, permutation tests, noise ceiling estimation,
and multiple comparison correction.
"""

import logging

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def bootstrap_ci(
    data: np.ndarray,
    statistic_fn=np.mean,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """Compute bootstrap confidence intervals for an arbitrary statistic.

    Args:
        data: 1D array of values
        statistic_fn: function to compute statistic from resampled data
        n_bootstrap: number of bootstrap samples
        confidence_level: CI level
        seed: random seed

    Returns:
        dict with 'observed', 'ci_low', 'ci_high', 'se'
    """
    rng = np.random.RandomState(seed)
    observed = statistic_fn(data)
    n = len(data)

    boot_stats = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_stats[b] = statistic_fn(data[idx])

    alpha = 1 - confidence_level
    return {
        "observed": float(observed),
        "ci_low": float(np.percentile(boot_stats, 100 * alpha / 2)),
        "ci_high": float(np.percentile(boot_stats, 100 * (1 - alpha / 2))),
        "se": float(np.std(boot_stats)),
    }


def permutation_test(
    group_a: np.ndarray,
    group_b: np.ndarray,
    statistic: str = "mean_diff",
    n_permutations: int = 5000,
    alternative: str = "two-sided",
    seed: int = 42,
) -> dict:
    """Two-sample permutation test.

    Args:
        group_a, group_b: arrays of values to compare
        statistic: 'mean_diff' or 'rank_sum'
        n_permutations: number of permutations
        alternative: 'two-sided', 'greater', or 'less'
        seed: random seed

    Returns:
        dict with 'observed', 'p_value', 'null_distribution'
    """
    rng = np.random.RandomState(seed)

    if statistic == "mean_diff":
        stat_fn = lambda a, b: np.mean(a) - np.mean(b)
    elif statistic == "rank_sum":
        stat_fn = lambda a, b: stats.rankdata(np.concatenate([a, b]))[:len(a)].sum()
    else:
        raise ValueError(f"Unknown statistic: {statistic}")

    observed = stat_fn(group_a, group_b)
    combined = np.concatenate([group_a, group_b])
    n_a = len(group_a)

    null_values = np.zeros(n_permutations)
    for p in range(n_permutations):
        perm = rng.permutation(len(combined))
        perm_a = combined[perm[:n_a]]
        perm_b = combined[perm[n_a:]]
        null_values[p] = stat_fn(perm_a, perm_b)

    if alternative == "two-sided":
        p_value = np.mean(np.abs(null_values) >= np.abs(observed))
    elif alternative == "greater":
        p_value = np.mean(null_values >= observed)
    elif alternative == "less":
        p_value = np.mean(null_values <= observed)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    return {
        "observed": float(observed),
        "p_value": float(p_value),
        "null_distribution": null_values,
    }


def fdr_correction(p_values: np.ndarray, alpha: float = 0.05) -> dict:
    """Benjamini-Hochberg FDR correction for multiple comparisons.

    Args:
        p_values: array of p-values
        alpha: significance threshold

    Returns:
        dict with:
            'reject': boolean array indicating which hypotheses to reject
            'adjusted_p': FDR-adjusted p-values
            'threshold': significance threshold after correction
    """
    p_flat = np.asarray(p_values).flatten()
    n = len(p_flat)
    sorted_idx = np.argsort(p_flat)
    sorted_p = p_flat[sorted_idx]

    # BH procedure
    adjusted = np.zeros(n)
    cummin = 1.0
    for i in range(n - 1, -1, -1):
        adjusted_val = sorted_p[i] * n / (i + 1)
        cummin = min(cummin, adjusted_val)
        adjusted[sorted_idx[i]] = min(cummin, 1.0)

    reject = adjusted <= alpha

    # Find threshold: largest p-value that is still rejected
    rejected_p = sorted_p[reject[sorted_idx]]
    threshold = float(rejected_p[-1]) if len(rejected_p) > 0 else 0.0

    return {
        "reject": reject.reshape(p_values.shape) if hasattr(p_values, 'shape') else reject,
        "adjusted_p": adjusted.reshape(p_values.shape) if hasattr(p_values, 'shape') else adjusted,
        "threshold": threshold,
        "n_rejected": int(reject.sum()),
        "n_total": n,
    }


def noise_ceiling_encoding(
    voxels_per_subject: list[np.ndarray],
) -> dict:
    """Compute noise ceiling for encoding models.

    Upper bound: average inter-subject correlation (each subject predicted
    by the group mean including that subject).
    Lower bound: average LOO correlation (each subject predicted by the
    group mean excluding that subject).

    Args:
        voxels_per_subject: list of (n_stimuli, n_voxels) arrays

    Returns:
        dict with 'upper', 'lower' bounds as mean per-voxel correlations
    """
    n_subjects = len(voxels_per_subject)
    if n_subjects < 2:
        return {"upper": np.nan, "lower": np.nan}

    n_stimuli = min(v.shape[0] for v in voxels_per_subject)
    n_voxels = voxels_per_subject[0].shape[1]
    aligned = np.stack([v[:n_stimuli] for v in voxels_per_subject])
    # aligned: (n_subjects, n_stimuli, n_voxels)

    group_mean = aligned.mean(axis=0)  # (n_stimuli, n_voxels)

    upper_rs = []
    lower_rs = []

    for i in range(n_subjects):
        subj_data = aligned[i]  # (n_stimuli, n_voxels)

        # Upper: correlate with full group mean
        voxel_rs = []
        for v in range(n_voxels):
            if np.std(subj_data[:, v]) < 1e-10:
                continue
            r, _ = stats.pearsonr(subj_data[:, v], group_mean[:, v])
            if np.isfinite(r):
                voxel_rs.append(r)
        upper_rs.append(np.mean(voxel_rs) if voxel_rs else 0.0)

        # Lower: correlate with LOO mean
        loo_mean = np.delete(aligned, i, axis=0).mean(axis=0)
        voxel_rs_loo = []
        for v in range(n_voxels):
            if np.std(subj_data[:, v]) < 1e-10:
                continue
            r, _ = stats.pearsonr(subj_data[:, v], loo_mean[:, v])
            if np.isfinite(r):
                voxel_rs_loo.append(r)
        lower_rs.append(np.mean(voxel_rs_loo) if voxel_rs_loo else 0.0)

    return {
        "upper": float(np.mean(upper_rs)),
        "lower": float(np.mean(lower_rs)),
        "per_subject_upper": np.array(upper_rs),
        "per_subject_lower": np.array(lower_rs),
    }


def test_dissociation(
    effects: np.ndarray,
    row_labels: list[str] = None,
    col_labels: list[str] = None,
) -> dict:
    """Test for a dissociation pattern in a conditions x measures matrix.

    Tests whether there is a significant interaction effect: each condition
    selectively impairs its corresponding measure more than other measures.

    Args:
        effects: (n_conditions, n_measures) matrix, e.g. (3 stages, 3 task types)
                 values represent performance DROP (larger = more impairment)
        row_labels: labels for rows (patched stages)
        col_labels: labels for columns (task types)

    Returns:
        dict with interaction statistics and selectivity indices
    """
    n_rows, n_cols = effects.shape

    # Selectivity index for each diagonal element:
    # How much more the diagonal is affected compared to off-diagonal
    selectivity = np.zeros(min(n_rows, n_cols))
    for i in range(min(n_rows, n_cols)):
        diagonal_val = effects[i, i]
        off_diag = np.delete(effects[i], i)
        selectivity[i] = diagonal_val - np.mean(off_diag)

    # Test interaction with two-way ANOVA-like analysis
    # Using Friedman test as a nonparametric alternative
    row_means = effects.mean(axis=1)
    col_means = effects.mean(axis=0)
    grand_mean = effects.mean()

    # Interaction effects: deviations from additive model
    interaction = np.zeros_like(effects)
    for i in range(n_rows):
        for j in range(n_cols):
            interaction[i, j] = (effects[i, j] - row_means[i] - col_means[j]
                                 + grand_mean)

    # Is the diagonal dominant? Compare diagonal vs off-diagonal
    diag_vals = np.diag(effects[:min(n_rows, n_cols), :min(n_rows, n_cols)])
    off_diag_mask = ~np.eye(min(n_rows, n_cols), dtype=bool)
    off_diag_vals = effects[:min(n_rows, n_cols), :min(n_rows, n_cols)][off_diag_mask]

    # Wilcoxon test for diagonal > off-diagonal
    if len(diag_vals) >= 3 and len(off_diag_vals) >= 3:
        stat, p_wilcox = stats.mannwhitneyu(diag_vals, off_diag_vals,
                                             alternative='greater')
    else:
        stat, p_wilcox = np.nan, np.nan

    return {
        "selectivity_indices": selectivity,
        "mean_selectivity": float(np.mean(selectivity)),
        "interaction_matrix": interaction,
        "diagonal_mean": float(np.mean(diag_vals)),
        "off_diagonal_mean": float(np.mean(off_diag_vals)),
        "diagonal_vs_offdiag_p": float(p_wilcox),
        "diagonal_vs_offdiag_stat": float(stat) if np.isfinite(stat) else None,
        "row_labels": row_labels,
        "col_labels": col_labels,
    }


def test_convergence_across_models(
    alignment_matrices: list[np.ndarray],
    model_names: list[str],
) -> dict:
    """Test whether stage-brain alignment patterns converge across models.

    Computes pairwise Spearman correlations between alignment profiles.

    Args:
        alignment_matrices: list of (n_layers_i, n_rois) alignment matrices
        model_names: corresponding model names

    Returns:
        dict with pairwise correlations and significance
    """
    n_models = len(alignment_matrices)
    pairwise_rho = np.zeros((n_models, n_models))
    pairwise_p = np.zeros((n_models, n_models))

    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                pairwise_rho[i, j] = 1.0
                pairwise_p[i, j] = 0.0
                continue
            # Flatten alignment matrices for comparison
            # Normalize layer indices to [0, 1] range for cross-model comparison
            mat_i = alignment_matrices[i]
            mat_j = alignment_matrices[j]
            # Interpolate to same number of layers
            n_common = min(mat_i.shape[0], mat_j.shape[0])
            from scipy.interpolate import interp1d
            if mat_i.shape[0] != n_common:
                x_old = np.linspace(0, 1, mat_i.shape[0])
                x_new = np.linspace(0, 1, n_common)
                f = interp1d(x_old, mat_i, axis=0)
                mat_i = f(x_new)
            if mat_j.shape[0] != n_common:
                x_old = np.linspace(0, 1, mat_j.shape[0])
                x_new = np.linspace(0, 1, n_common)
                f = interp1d(x_old, mat_j, axis=0)
                mat_j = f(x_new)

            rho, p = stats.spearmanr(mat_i.flatten(), mat_j.flatten())
            pairwise_rho[i, j] = rho
            pairwise_p[i, j] = p

    return {
        "pairwise_rho": pairwise_rho,
        "pairwise_p": pairwise_p,
        "model_names": model_names,
        "mean_rho": float(pairwise_rho[np.triu_indices(n_models, k=1)].mean()),
    }
