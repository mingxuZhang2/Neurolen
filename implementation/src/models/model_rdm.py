"""Compute Representational Dissimilarity Matrices from MLLM activation arrays."""

import logging
from pathlib import Path

import numpy as np
import rsatoolbox
from tqdm import tqdm

logger = logging.getLogger(__name__)


def compute_rdm(
    activations: np.ndarray,
    method: str = "correlation",
) -> rsatoolbox.rdm.RDMs:
    """Compute an RDM from an activation matrix.

    Args:
        activations: (n_stimuli, n_features) activation matrix
        method: distance metric, one of 'correlation', 'euclidean', 'mahalanobis'

    Returns:
        rsatoolbox RDMs object
    """
    dataset = rsatoolbox.data.Dataset(activations)
    rdm = rsatoolbox.rdm.calc_rdm(dataset, method=method)
    return rdm


def compute_rdm_raw(
    activations: np.ndarray,
    method: str = "correlation",
) -> np.ndarray:
    """Compute RDM as a raw numpy dissimilarity matrix.

    Args:
        activations: (n_stimuli, n_features)
        method: 'correlation' for 1 - Pearson r, 'euclidean' for Euclidean distance

    Returns:
        (n_stimuli, n_stimuli) dissimilarity matrix
    """
    n = activations.shape[0]
    if method == "correlation":
        # Center each stimulus vector
        centered = activations - activations.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # avoid division by zero
        normed = centered / norms
        corr_matrix = normed @ normed.T
        # Clip to [-1, 1] for numerical stability
        corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
        rdm_matrix = 1.0 - corr_matrix
    elif method == "euclidean":
        from scipy.spatial.distance import pdist, squareform
        rdm_matrix = squareform(pdist(activations, metric="euclidean"))
    else:
        raise ValueError(f"Unknown method: {method}")
    return rdm_matrix


def compute_model_rdms(
    activations_dir: str,
    model_name: str,
    n_layers: int,
    token_types: list[str] = None,
    method: str = "correlation",
    output_dir: str = None,
) -> dict[str, rsatoolbox.rdm.RDMs]:
    """Compute RDMs for all layers and token types of a model.

    Args:
        activations_dir: directory containing activation numpy files
        model_name: model identifier (subdirectory name)
        n_layers: number of decoder layers
        token_types: list of token types to process, default ["vis", "txt", "all"]
        method: distance metric
        output_dir: if provided, save RDMs to disk

    Returns:
        dict mapping "layer_{l}_{token_type}" -> RDMs object
    """
    if token_types is None:
        token_types = ["vis", "txt", "all"]

    act_dir = Path(activations_dir) / model_name
    rdms = {}

    for l in tqdm(range(n_layers), desc=f"Computing RDMs for {model_name}"):
        for tt in token_types:
            key = f"layer_{l}_{tt}"
            fpath = act_dir / f"{key}.npy"
            if not fpath.exists():
                logger.warning(f"Missing: {fpath}")
                continue
            data = np.load(fpath)
            if data.shape[0] < 3:
                logger.warning(f"Too few stimuli ({data.shape[0]}) for RDM: {key}")
                continue
            rdm = compute_rdm(data, method=method)
            rdms[key] = rdm

    if output_dir:
        out_path = Path(output_dir) / model_name
        out_path.mkdir(parents=True, exist_ok=True)
        for key, rdm in rdms.items():
            # Save the upper triangle as a flat vector
            np.save(out_path / f"rdm_{key}.npy", rdm.get_vectors())
            # Also save the full matrix for convenience
            n = int(0.5 + np.sqrt(0.25 + 2 * rdm.get_vectors().shape[1]))
            from scipy.spatial.distance import squareform
            full_mat = squareform(rdm.get_vectors().flatten())
            np.save(out_path / f"rdm_matrix_{key}.npy", full_mat)
        logger.info(f"Saved {len(rdms)} RDMs to {out_path}")

    return rdms


def load_model_rdm(rdm_dir: str, model_name: str, layer: int,
                   token_type: str = "all") -> np.ndarray:
    """Load a precomputed RDM matrix from disk."""
    path = Path(rdm_dir) / model_name / f"rdm_matrix_layer_{layer}_{token_type}.npy"
    if not path.exists():
        raise FileNotFoundError(f"RDM not found: {path}")
    return np.load(path)


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Compute model RDMs")
    parser.add_argument("--model", required=True, choices=["llava", "qwen2vl", "internvl2"])
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--activations-dir", default="activations")
    parser.add_argument("--output-dir", default="rdms")
    parser.add_argument("--method", default="correlation")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    with open(args.config) as f:
        config = yaml.safe_load(f)
    model_cfg = config["models"][args.model]

    # Clean model name to match directory structure
    model_dir_name = model_cfg["name"].replace(" ", "_").replace("-", "_")

    compute_model_rdms(
        activations_dir=args.activations_dir,
        model_name=model_dir_name,
        n_layers=model_cfg["n_layers"],
        method=args.method,
        output_dir=args.output_dir,
    )
