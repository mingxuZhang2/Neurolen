"""NeuroLens token-level pipeline orchestrator.

Steps:
  extract  → run TokenLevelExtractor on shared1000, save per-layer token activations + logit lens
  spatial  → fit Ridge for each (token, layer) → V1..V4 voxels, save best-voxel maps
  semantic → build semantic vectors from logit-lens top-K words, fit Ridge → language ROIs
  visualize → produce F1-F4 figure series

Each step is idempotent and saves intermediate outputs so it can resume.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("neurolens_token")


def _add_project_to_path():
    here = Path(__file__).resolve()
    impl = here.parent.parent
    sys.path.insert(0, str(impl))


_add_project_to_path()


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def step_extract(args):
    """Run TokenLevelExtractor on shared1000 stimuli."""
    from src.models.token_extract import TokenLevelExtractor

    models_cfg = _load_yaml(Path(args.config_dir) / "models.yaml")
    model_cfg = dict(models_cfg["models"][args.model])
    if args.model_path_override:
        model_cfg["hf_id"] = args.model_path_override

    out_dir = Path(args.output_root) / "token_activations" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.output_root) / "stimuli_shared1000" / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    image_paths = [
        str(Path(args.output_root) / "stimuli_shared1000" / p)
        for p in manifest["paths"]
    ]
    if args.limit and args.limit > 0:
        image_paths = image_paths[: args.limit]
        manifest["nsd_ids"] = manifest["nsd_ids"][: args.limit]
    nsd_ids = manifest["nsd_ids"]
    prompts = ["Describe this image in detail."] * len(image_paths)

    extractor = TokenLevelExtractor(model_cfg, device="cuda", dtype="float16",
                                     top_k=args.top_k)
    extractor.load_model()
    extractor.extract_token_level_all(
        images=image_paths,
        texts=prompts,
        output_dir=str(out_dir),
        pca_dim=args.pca_dim,
        save_raw=False,
        log_every=20,
    )
    extractor.cleanup()

    # Save NSD ordering
    model_subdir = model_cfg["name"].replace(" ", "_").replace("-", "_")
    actual = out_dir / model_subdir
    for target in (out_dir, actual):
        with open(target / "nsd_ids.json", "w") as f:
            json.dump(nsd_ids, f)
    logger.info(f"Token-level extraction complete: {actual}")


def step_clip_baseline(args):
    """Extract CLIP patch tokens + projector output as baselines."""
    from src.models.clip_baseline_extract import extract_clip_baselines

    models_cfg = _load_yaml(Path(args.config_dir) / "models.yaml")
    model_cfg = models_cfg["models"][args.model]
    llava_dir = args.model_path_override or model_cfg["hf_id"]

    manifest_path = Path(args.output_root) / "stimuli_shared1000" / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    image_paths = [
        str(Path(args.output_root) / "stimuli_shared1000" / p)
        for p in manifest["paths"]
    ]
    if args.limit and args.limit > 0:
        image_paths = image_paths[: args.limit]

    out_dir = Path(args.output_root) / "token_activations" / "clip_baseline"
    extract_clip_baselines(
        llava_dir=llava_dir,
        image_paths=image_paths,
        output_dir=str(out_dir),
        pca_dim=args.pca_dim,
    )

    # Copy nsd_ids from main extractor
    model_subdir = model_cfg["name"].replace(" ", "_").replace("-", "_")
    src_nsd = Path(args.output_root) / "token_activations" / args.model / model_subdir / "nsd_ids.json"
    if src_nsd.exists():
        import shutil
        shutil.copy(src_nsd, out_dir / "nsd_ids.json")
    logger.info(f"CLIP baseline extraction done: {out_dir}")


def step_spatial(args):
    """Token × layer → V1..V4 spatial encoding."""
    from src.analysis.token_encoding import (
        run_token_spatial_encoding, load_token_activations,
    )
    from src.data.nsd_fsaverage import (
        NSDFsaverageLoader, get_roi_voxels,
    )

    models_cfg = _load_yaml(Path(args.config_dir) / "models.yaml")
    model_cfg = models_cfg["models"][args.model]
    model_subdir = model_cfg["name"].replace(" ", "_").replace("-", "_")

    act_dir = Path(args.output_root) / "token_activations" / args.model / model_subdir
    if not act_dir.exists():
        act_dir = Path(args.output_root) / "token_activations" / args.model
    nsd_ids_file = act_dir / "nsd_ids.json"
    with open(nsd_ids_file) as f:
        activation_nsd_ids = np.array(json.load(f))

    layers_to_run = args.layers
    token_acts = load_token_activations(act_dir, layers_to_run)
    if not token_acts:
        raise RuntimeError(f"No token activations found in {act_dir}")
    logger.info(f"Loaded token activations for layers {sorted(token_acts.keys())}")

    loader = NSDFsaverageLoader(args.nsd_root)
    subj = args.subjects[0] if args.subjects else 1
    visual_rois = args.visual_rois or ["V1", "V2", "V3", "V4"]
    brain_per_roi = {}
    for roi in visual_rois:
        try:
            voxels, ids, ncsnr = get_roi_voxels(
                loader, subj - 1, roi, max_sessions=args.max_sessions,
            )
        except Exception as e:
            logger.warning(f"ROI {roi} failed: {e}")
            continue
        if voxels.shape[1] == 0:
            continue
        brain_per_roi[roi] = {"voxels": voxels, "nsd_ids": ids, "ncsnr": ncsnr}

    out_root = Path(args.output_root) / "token_spatial_real" / f"subj{subj:02d}"
    run_token_spatial_encoding(
        token_activations_per_layer=token_acts,
        activation_nsd_ids=activation_nsd_ids,
        brain_per_roi=brain_per_roi,
        n_folds=args.n_folds,
        output_dir=out_root,
    )


def step_semantic(args):
    """Token × layer logit-lens → language ROI semantic encoding."""
    from src.analysis.token_encoding import (
        run_token_semantic_encoding, load_logit_lens,
    )
    from src.data.nsd_fsaverage import (
        NSDFsaverageLoader, get_roi_voxels,
    )

    models_cfg = _load_yaml(Path(args.config_dir) / "models.yaml")
    model_cfg = models_cfg["models"][args.model]
    model_subdir = model_cfg["name"].replace(" ", "_").replace("-", "_")

    act_dir = Path(args.output_root) / "token_activations" / args.model / model_subdir
    if not act_dir.exists():
        act_dir = Path(args.output_root) / "token_activations" / args.model
    nsd_ids_file = act_dir / "nsd_ids.json"
    with open(nsd_ids_file) as f:
        activation_nsd_ids = np.array(json.load(f))

    logger.info("Loading logit lens arrays...")
    ids, probs = load_logit_lens(act_dir, args.layers)
    logger.info(f"ids shape={ids.shape}, probs shape={probs.shape}")

    # Load brain ROIs FIRST while memory is still free.
    # (The LLaVA shard load below will hold ~10 GB.)
    loader_brain = NSDFsaverageLoader(args.nsd_root)
    subj = args.subjects[0] if args.subjects else 1
    language_rois = args.language_rois or ["Broca", "IFG_extended", "auditory_assoc", "temporal_pole"]
    brain_per_roi = {}
    for roi in language_rois:
        try:
            voxels, ids_r, ncsnr = get_roi_voxels(
                loader_brain, subj - 1, roi, max_sessions=args.max_sessions,
            )
        except Exception as e:
            logger.warning(f"ROI {roi} failed: {e}")
            continue
        if voxels.shape[1] == 0:
            continue
        brain_per_roi[roi] = {"voxels": voxels, "nsd_ids": ids_r, "ncsnr": ncsnr}

    import gc
    gc.collect()

    # Word embedding table: load embed_tokens directly from a single shard.
    # Release the shard dict immediately after extracting the one tensor we need.
    logger.info("Loading word embedding table directly from LLaVA shard...")
    import torch
    import os
    llava_dir = args.model_path_override or model_cfg["hf_id"]
    word_emb_table = None
    for shard_name in ("pytorch_model-00001-of-00002.bin", "pytorch_model-00002-of-00002.bin",
                        "pytorch_model.bin"):
        sp = os.path.join(llava_dir, shard_name)
        if not os.path.exists(sp):
            continue
        shard = torch.load(sp, map_location="cpu", weights_only=True)
        for k, v in shard.items():
            if "embed_tokens.weight" in k and "vision" not in k:
                word_emb_table = v.to(torch.float32).numpy()
                logger.info(f"  Found {k} in {shard_name}: shape {word_emb_table.shape}")
                break
        del shard
        gc.collect()
        if word_emb_table is not None:
            break
    if word_emb_table is None:
        raise RuntimeError("Could not find embed_tokens.weight in LLaVA shards")
    logger.info(f"word_emb_table loaded: {word_emb_table.shape}")

    out_root = Path(args.output_root) / "token_semantic_real" / f"subj{subj:02d}"
    run_token_semantic_encoding(
        logit_ids=ids,
        logit_probs=probs,
        word_emb_table=word_emb_table,
        activation_nsd_ids=activation_nsd_ids,
        brain_per_roi=brain_per_roi,
        n_folds=args.n_folds,
        sem_pca_dim=args.pca_dim,
        output_dir=out_root,
    )


def step_visualize(args):
    """Produce F1-F4 figures from token-level outputs."""
    from src.visualization.token_maps import (
        plot_token_retinotopy_on_voxels,
        plot_retinotopy_local_coherence,
        plot_logit_lens_words,
        plot_spatial_semantic_transition,
        plot_brain_hierarchy_match,
    )
    from src.analysis.token_encoding import load_logit_lens

    models_cfg = _load_yaml(Path(args.config_dir) / "models.yaml")
    model_cfg = models_cfg["models"][args.model]
    model_subdir = model_cfg["name"].replace(" ", "_").replace("-", "_")
    act_dir = Path(args.output_root) / "token_activations" / args.model / model_subdir
    if not act_dir.exists():
        act_dir = Path(args.output_root) / "token_activations" / args.model

    subj = args.subjects[0] if args.subjects else 1
    fig_dir = Path(args.output_root) / "figures_token" / f"subj{subj:02d}"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load spatial encoding
    spat_path = (Path(args.output_root) / "token_spatial_real" /
                  f"subj{subj:02d}" / "spatial_encoding.npz")
    if not spat_path.exists():
        logger.warning(f"Missing {spat_path}; skip F1/F4")
    else:
        spat = np.load(spat_path, allow_pickle=True)
        layer_indices = spat["layer_indices"]
        best_voxel = spat["best_voxel"]
        spat_best_r = spat["best_r"]
        vis_rois = list(spat["roi_names"])

        # F1: retinotopy at 2-3 representative layers
        if len(layer_indices) > 0:
            for L_pos in (0, min(8, len(layer_indices) - 1), len(layer_indices) - 1):
                L = int(layer_indices[L_pos])
                for r_idx, r_name in enumerate(vis_rois):
                    plot_token_retinotopy_on_voxels(
                        best_voxel=best_voxel,
                        layer_idx=L,
                        roi_name=r_name,
                        roi_idx=r_idx,
                        layer_indices=layer_indices,
                        out_path=fig_dir / f"F1_retinotopy_L{L}_{r_name}.png",
                    )
        plot_retinotopy_local_coherence(
            best_voxel=best_voxel,
            roi_names=vis_rois,
            layer_indices=layer_indices,
            out_path=fig_dir / "F1b_retinotopy_coherence.png",
        )

    # Load semantic encoding
    sem_path = (Path(args.output_root) / "token_semantic_real" /
                f"subj{subj:02d}" / "semantic_encoding.npz")
    if not sem_path.exists():
        logger.warning(f"Missing {sem_path}; skip F3 punch-line")
    elif spat_path.exists():
        sem = np.load(sem_path, allow_pickle=True)
        sem_best_r = sem["best_r"]      # (n_tokens, n_layers, n_lang_rois)
        lang_rois = list(sem["roi_names"])

        # Use mean_r (unbiased) instead of best_r for F3/F4
        spat_mean_r = spat["mean_r"]   # (n_tokens, n_layers, n_vis_rois)
        sem_mean_r = sem["mean_r"]     # (n_tokens, n_layers, n_lang_rois)
        # Reduce to scalar per (token, layer): mean across ROIs
        spat_max = spat_mean_r.mean(axis=2)
        sem_max = sem_mean_r.mean(axis=2)

        plot_spatial_semantic_transition(
            spatial_r=spat_max,
            semantic_r=sem_max,
            layer_indices=spat["layer_indices"],
            out_path=fig_dir / "F3_spatial_semantic.png",
        )
        plot_brain_hierarchy_match(
            spatial_per_roi=spat_mean_r,
            semantic_per_roi=sem_mean_r,
            visual_rois=vis_rois,
            language_rois=lang_rois,
            layer_indices=spat["layer_indices"],
            out_path=fig_dir / "F4_brain_hierarchy.png",
        )

    # F2: logit lens trajectory (one example image)
    vocab_inv_path = act_dir / "vocab_inv.json"
    if vocab_inv_path.exists():
        with open(vocab_inv_path) as f:
            vocab_inv = json.load(f)
        ids, probs = load_logit_lens(act_dir, args.layers)
        plot_logit_lens_words(
            logit_ids=ids,
            logit_probs=probs,
            vocab_inv=vocab_inv,
            out_path=fig_dir / "F2_logit_lens_words.png",
            n_example_layers=6,
            n_example_tokens=12,
            image_idx=args.image_idx,
        )
    else:
        logger.warning(f"Missing vocab_inv at {vocab_inv_path}; skip F2")

    logger.info(f"Saved figures to {fig_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--step", required=True,
                   choices=["extract", "clip_baseline", "spatial", "semantic",
                            "visualize", "all"])
    p.add_argument("--model", default="llava")
    p.add_argument("--model-path-override", default=None)
    p.add_argument("--nsd-root", default="/hpc2hdd/home/mzhang630/data/nsd")
    p.add_argument("--output-root",
                   default=os.environ.get("PROJECT_DIR",
                                          "/hpc2hdd/home/mzhang630/data/mllm") + "/results_v2")
    p.add_argument("--config-dir", default=str(
        Path(__file__).resolve().parent.parent / "configs"))
    p.add_argument("--subjects", type=int, nargs="*", default=[1])
    p.add_argument("--max-sessions", type=int, default=None)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--pca-dim", type=int, default=256)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--layers", type=int, nargs="*", default=None,
                   help="Subset of layers; default: all available")
    p.add_argument("--visual-rois", nargs="*", default=None)
    p.add_argument("--language-rois", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, subsample first N stimuli (smoke test)")
    p.add_argument("--image-idx", type=int, default=0,
                   help="Which image to use for F2 logit-lens visualization")
    return p.parse_args()


def main():
    args = parse_args()
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    steps = [args.step] if args.step != "all" else [
        "extract", "spatial", "semantic", "visualize"
    ]
    dispatch = {
        "extract": step_extract,
        "clip_baseline": step_clip_baseline,
        "spatial": step_spatial,
        "semantic": step_semantic,
        "visualize": step_visualize,
    }
    for s in steps:
        logger.info(f"\n========== TOKEN STEP: {s} ==========")
        dispatch[s](args)
        logger.info(f"========== STEP {s} done ==========\n")


if __name__ == "__main__":
    main()
