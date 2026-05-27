"""NeuroLens: Main experiment orchestrator.

Maps MLLM processing stages to brain functional hierarchy.
Runs the full pipeline: extract -> analyze -> patch -> visualize.

Usage:
    python run_experiment.py --step extract --model llava
    python run_experiment.py --step analyze --model llava
    python run_experiment.py --step patch --model llava
    python run_experiment.py --step visualize --model llava
    python run_experiment.py --step all --model llava
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger("neurolens")

# Project root: the directory containing this script
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


def load_configs(args) -> dict:
    """Load all configuration files."""
    config_dir = PROJECT_ROOT / "configs"
    with open(config_dir / "models.yaml") as f:
        models_config = yaml.safe_load(f)
    with open(config_dir / "rois.yaml") as f:
        rois_config = yaml.safe_load(f)
    with open(config_dir / "experiments.yaml") as f:
        exp_config = yaml.safe_load(f)

    # Override with CLI arguments
    if args.nsd_root:
        exp_config["paths"]["nsd_root"] = args.nsd_root
    elif os.environ.get("NSD_ROOT"):
        exp_config["paths"]["nsd_root"] = os.environ["NSD_ROOT"]
    if args.coco_root:
        exp_config["paths"]["coco_root"] = args.coco_root
    elif os.environ.get("COCO_ROOT"):
        exp_config["paths"]["coco_root"] = os.environ["COCO_ROOT"]
    if args.output_dir:
        exp_config["paths"]["results_dir"] = args.output_dir

    return {
        "models": models_config,
        "rois": rois_config,
        "experiments": exp_config,
    }


def step_extract(args, config: dict):
    """Step 1: Extract MLLM activations for all stimuli."""
    from src.data.nsd_loader import NSDLoader
    from src.data.stimulus_align import StimulusAligner
    from src.models.extract_activations import ActivationExtractor, load_model_config

    exp = config["experiments"]
    nsd_root = exp["paths"]["nsd_root"]
    coco_root = exp["paths"]["coco_root"]
    act_dir = PROJECT_ROOT / exp["paths"]["activations_dir"]

    # Load NSD shared images
    nsd = NSDLoader(nsd_root, str(PROJECT_ROOT / "configs" / "rois.yaml"))
    shared_ids = nsd.get_shared_image_ids()
    logger.info(f"Found {len(shared_ids)} shared NSD images")

    # Align with COCO
    aligner = StimulusAligner(nsd_root, coco_root)
    stimulus_file = PROJECT_ROOT / "stimuli" / "stimulus_pairs.json"

    if stimulus_file.exists():
        pairs = aligner.load_stimulus_pairs(str(stimulus_file))
        logger.info(f"Loaded existing stimulus pairs: {len(pairs['image_paths'])} images")
    else:
        prompt = exp["extraction"]["prompt_template"]
        pairs = aligner.create_stimulus_pairs(
            shared_ids, prompt_template=prompt,
            use_coco_captions=True,
        )
        aligner.save_stimulus_pairs(pairs, str(stimulus_file))

    # Extract activations for specified model
    models_to_run = [args.model] if args.model != "all" else exp["extraction"]["models"]

    for model_key in models_to_run:
        logger.info(f"\n{'='*60}")
        logger.info(f"Extracting activations for: {model_key}")
        logger.info(f"{'='*60}")

        model_config = config["models"]["models"][model_key]
        extractor = ActivationExtractor(model_config)
        extractor.load_model()

        extractor.extract_all(
            images=pairs["image_paths"],
            texts=pairs["texts"],
            output_dir=str(act_dir),
            pca_dim=exp["extraction"]["pca_dim"],
        )

        if exp["extraction"]["extract_vision_encoder"]:
            extractor.extract_vision_encoder(
                images=pairs["image_paths"],
                output_dir=str(act_dir),
            )

        extractor.cleanup()
        logger.info(f"Completed extraction for {model_key}")


def step_brain_data(args, config: dict):
    """Step 1b: Process NSD brain data and compute brain RDMs."""
    from src.data.nsd_loader import NSDLoader

    exp = config["experiments"]
    nsd_root = exp["paths"]["nsd_root"]
    brain_dir = PROJECT_ROOT / exp["paths"]["brain_data_dir"]

    nsd = NSDLoader(nsd_root, str(PROJECT_ROOT / "configs" / "rois.yaml"))
    shared_ids = nsd.get_shared_image_ids()

    nsd.compute_all_brain_rdms(
        subjects=exp["brain_data"]["subjects"],
        roi_names=exp["brain_data"]["roi_groups"] + exp["brain_data"]["individual_rois"],
        nsd_ids=shared_ids,
        output_dir=str(brain_dir),
        method=exp["brain_data"]["rdm_metric"],
    )
    logger.info("Brain data processing complete")


def step_model_rdms(args, config: dict):
    """Step 2a: Compute model RDMs from extracted activations."""
    from src.models.model_rdm import compute_model_rdms

    exp = config["experiments"]
    act_dir = str(PROJECT_ROOT / exp["paths"]["activations_dir"])
    rdm_dir = str(PROJECT_ROOT / "rdms")

    models_to_run = [args.model] if args.model != "all" else exp["extraction"]["models"]

    for model_key in models_to_run:
        model_cfg = config["models"]["models"][model_key]
        model_name = model_cfg["name"].replace(" ", "_").replace("-", "_")

        logger.info(f"Computing model RDMs for {model_key}")
        compute_model_rdms(
            activations_dir=act_dir,
            model_name=model_name,
            n_layers=model_cfg["n_layers"],
            method=exp["brain_data"]["rdm_metric"],
            output_dir=rdm_dir,
        )


def step_analyze(args, config: dict):
    """Step 3: Run RSA, CKA, and encoding model analyses."""
    from src.analysis.rsa_analysis import run_full_rsa
    from src.analysis.cka_analysis import compute_cka_matrix
    from src.analysis.encoding_models import compute_encoding_matrix

    exp = config["experiments"]
    act_dir = str(PROJECT_ROOT / exp["paths"]["activations_dir"])
    brain_dir = str(PROJECT_ROOT / exp["paths"]["brain_data_dir"])
    results_dir = str(PROJECT_ROOT / exp["paths"]["results_dir"])
    rdm_dir = str(PROJECT_ROOT / "rdms")

    subjects = exp["brain_data"]["subjects"]
    rois = exp["brain_data"]["roi_groups"]

    models_to_run = [args.model] if args.model != "all" else exp["extraction"]["models"]

    for model_key in models_to_run:
        model_cfg = config["models"]["models"][model_key]
        model_name = model_cfg["name"].replace(" ", "_").replace("-", "_")

        logger.info(f"\n{'='*60}")
        logger.info(f"Running analysis for: {model_key}")
        logger.info(f"{'='*60}")

        for token_type in ["vis", "txt", "all"]:
            logger.info(f"\n--- Token type: {token_type} ---")

            # RSA
            logger.info("Running RSA...")
            run_full_rsa(
                model_rdm_dir=rdm_dir,
                brain_rdm_dir=brain_dir,
                model_name=model_name,
                n_layers=model_cfg["n_layers"],
                subjects=subjects,
                roi_names=rois,
                token_type=token_type,
                method=exp["rsa"]["comparison_method"],
                n_bootstrap=exp["rsa"]["n_bootstrap"],
                n_permutations=exp["rsa"]["permutation_n"],
                output_dir=f"{results_dir}/rsa",
            )

            # CKA
            logger.info("Running CKA...")
            compute_cka_matrix(
                activations_dir=act_dir,
                brain_data_dir=brain_dir,
                model_name=model_name,
                n_layers=model_cfg["n_layers"],
                subjects=subjects,
                roi_names=rois,
                token_type=token_type,
                method=exp["cka"]["method"],
                n_bootstrap=exp["cka"]["n_bootstrap"],
                output_dir=f"{results_dir}/cka",
            )

            # Encoding models
            logger.info("Running encoding models...")
            compute_encoding_matrix(
                activations_dir=act_dir,
                brain_data_dir=brain_dir,
                model_name=model_name,
                n_layers=model_cfg["n_layers"],
                subjects=subjects,
                roi_names=rois,
                token_type=token_type,
                n_folds=exp["encoding"]["n_folds"],
                alphas=exp["encoding"]["alphas"],
                output_dir=f"{results_dir}/encoding",
            )


def step_patch(args, config: dict):
    """Step 4: Run activation patching and triple dissociation."""
    from src.models.extract_activations import ActivationExtractor
    from src.patching.dissociation import run_triple_dissociation

    exp = config["experiments"]
    results_dir = str(PROJECT_ROOT / exp["paths"]["results_dir"])

    models_to_run = [args.model] if args.model != "all" else exp["patching"]["models"]

    for model_key in models_to_run:
        model_cfg = config["models"]["models"][model_key]

        logger.info(f"\n{'='*60}")
        logger.info(f"Running patching for: {model_key}")
        logger.info(f"{'='*60}")

        # Load model
        extractor = ActivationExtractor(model_cfg)
        extractor.load_model()

        for patch_type in exp["patching"]["patch_types"]:
            logger.info(f"\n--- Patch type: {patch_type} ---")

            output_dir = f"{results_dir}/dissociation/{model_key}_{patch_type}"

            # Load stimulus pairs for mean activation computation
            stimulus_file = PROJECT_ROOT / "stimuli" / "stimulus_pairs.json"
            if stimulus_file.exists():
                with open(stimulus_file) as f:
                    pairs = json.load(f)
                mean_images = pairs["image_paths"][:100]
                mean_texts = pairs["texts"][:100]
            else:
                mean_images = None
                mean_texts = None

            run_triple_dissociation(
                model_config=model_cfg,
                model=extractor.model,
                processor=extractor.processor,
                image_dir=str(PROJECT_ROOT / "eval_images"),
                task_file=str(PROJECT_ROOT / "configs" / "dissociation_tasks.json"),
                n_samples=exp["dissociation"]["task_types"]["visual_recognition"]["n_samples"],
                patch_type=patch_type,
                mean_activation_images=mean_images,
                mean_activation_texts=mean_texts,
                output_dir=output_dir,
            )

        extractor.cleanup()


def step_visualize(args, config: dict):
    """Step 5: Generate all figures."""
    from src.visualization.heatmaps import (
        plot_alignment_heatmap, plot_multi_model_comparison,
        plot_dissociation_matrix, plot_layer_profile,
        plot_metric_comparison,
    )
    from src.visualization.brain_maps import (
        plot_glass_brain_alignment, plot_stage_brain_correspondence,
    )

    exp = config["experiments"]
    results_dir = PROJECT_ROOT / exp["paths"]["results_dir"]
    fig_dir = PROJECT_ROOT / exp["paths"]["figures_dir"]
    fig_dir.mkdir(parents=True, exist_ok=True)

    models_to_run = [args.model] if args.model != "all" else exp["extraction"]["models"]
    rois = exp["brain_data"]["roi_groups"]

    for model_key in models_to_run:
        model_cfg = config["models"]["models"][model_key]
        model_name = model_cfg["name"].replace(" ", "_").replace("-", "_")
        stages = model_cfg["stages"]
        boundaries = [stages["fusion"][0], stages["language"][0]]

        # Load results
        for token_type in ["all"]:
            # RSA heatmap
            rsa_file = results_dir / "rsa" / f"rsa_{model_name}_{token_type}.npz"
            if rsa_file.exists():
                data = np.load(rsa_file, allow_pickle=True)
                plot_alignment_heatmap(
                    matrix=data["rsa_matrix"],
                    roi_names=list(data["roi_names"]),
                    model_name=model_cfg["name"],
                    metric_name="RSA",
                    stage_boundaries=boundaries,
                    output_path=str(fig_dir / f"rsa_heatmap_{model_key}_{token_type}.pdf"),
                )

            # CKA heatmap
            cka_file = results_dir / "cka" / f"cka_{model_name}_{token_type}_linear.npz"
            if cka_file.exists():
                data = np.load(cka_file, allow_pickle=True)
                plot_alignment_heatmap(
                    matrix=data["cka_matrix"],
                    roi_names=list(data["roi_names"]),
                    model_name=model_cfg["name"],
                    metric_name="CKA",
                    stage_boundaries=boundaries,
                    output_path=str(fig_dir / f"cka_heatmap_{model_key}_{token_type}.pdf"),
                )

            # Encoding heatmap
            enc_file = results_dir / "encoding" / f"encoding_{model_name}_{token_type}.npz"
            if enc_file.exists():
                data = np.load(enc_file, allow_pickle=True)
                plot_alignment_heatmap(
                    matrix=data["encoding_matrix"],
                    roi_names=list(data["roi_names"]),
                    model_name=model_cfg["name"],
                    metric_name="Encoding (r)",
                    stage_boundaries=boundaries,
                    output_path=str(fig_dir / f"enc_heatmap_{model_key}_{token_type}.pdf"),
                )

        # Dissociation matrix
        for patch_type in exp["patching"]["patch_types"]:
            diss_file = (results_dir / "dissociation" /
                         f"{model_key}_{patch_type}" / "dissociation_matrix.npy")
            if diss_file.exists():
                matrix = np.load(diss_file)
                plot_dissociation_matrix(
                    matrix=matrix,
                    output_path=str(fig_dir / f"dissociation_{model_key}_{patch_type}.pdf"),
                )

    # Multi-model comparison (if multiple models available)
    if len(models_to_run) > 1:
        all_matrices = {}
        boundaries_map = {}
        for model_key in models_to_run:
            model_cfg = config["models"]["models"][model_key]
            model_name = model_cfg["name"].replace(" ", "_").replace("-", "_")
            rsa_file = results_dir / "rsa" / f"rsa_{model_name}_all.npz"
            if rsa_file.exists():
                data = np.load(rsa_file, allow_pickle=True)
                all_matrices[model_cfg["name"]] = data["rsa_matrix"]
                stages = model_cfg["stages"]
                boundaries_map[model_cfg["name"]] = [
                    stages["fusion"][0], stages["language"][0]
                ]

        if all_matrices:
            plot_multi_model_comparison(
                matrices=all_matrices,
                roi_names=rois,
                metric_name="RSA",
                stage_boundaries_per_model=boundaries_map,
                output_path=str(fig_dir / "multi_model_rsa.pdf"),
            )

    logger.info(f"All figures saved to {fig_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="NeuroLens: MLLM-Brain Alignment Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  extract     Extract MLLM activations for NSD stimuli
  brain       Process NSD brain data and compute brain RDMs
  rdms        Compute model RDMs from extracted activations
  analyze     Run RSA, CKA, and encoding model analyses
  patch       Run activation patching and triple dissociation
  visualize   Generate all figures
  all         Run the full pipeline
        """,
    )
    parser.add_argument("--step", required=True,
                        choices=["extract", "brain", "rdms", "analyze",
                                 "patch", "visualize", "all"])
    parser.add_argument("--model", default="llava",
                        choices=["llava", "qwen2vl", "internvl2", "all"])
    parser.add_argument("--nsd-root", default=None,
                        help="Path to NSD data root")
    parser.add_argument("--coco-root", default=None,
                        help="Path to COCO data root")
    parser.add_argument("--output-dir", default=None,
                        help="Override results output directory")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(PROJECT_ROOT / "experiment.log"),
        ],
    )

    config = load_configs(args)
    step = args.step

    STEPS = {
        "extract": step_extract,
        "brain": step_brain_data,
        "rdms": step_model_rdms,
        "analyze": step_analyze,
        "patch": step_patch,
        "visualize": step_visualize,
    }

    if step == "all":
        for step_name in ["extract", "brain", "rdms", "analyze", "patch", "visualize"]:
            logger.info(f"\n{'#'*60}")
            logger.info(f"# STEP: {step_name.upper()}")
            logger.info(f"{'#'*60}")
            STEPS[step_name](args, config)
    else:
        STEPS[step](args, config)

    logger.info("Done.")


if __name__ == "__main__":
    main()
