"""NeuroLens v2 pipeline — real NSD fMRI + encoding model + single-layer ablation.

Replaces v1's synthetic brain data and stage-level patching with:
  - Real NSD fsaverage betas (shared1000 averaged across 3 reps)
  - Real COCO-reconstructed NSD stimuli (via cropBox)
  - Ridge-regression encoding model with noise-ceiling normalization
  - Single-layer ablation sweep instead of 11-layer stages
  - Task-specific evaluation: object recognition / category-MC / caption fluency

Steps (selectable via --step):
  meta      → build COCO task metadata for shared1000
  stim      → reconstruct 1000 stimulus images from COCO + cropBox
  extract   → extract MLLM activations on shared1000
  brain     → load NSD betas, extract per-ROI shared1000 voxel patterns
  encode    → fit layer × ROI encoding models with noise ceiling
  sweep     → single-layer ablation × 3 tasks
  dissociate→ aggregate sweeps into stage × task dissociation matrix
  visualize → produce heatmaps
  all       → run everything
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("neurolens_v2")


def _add_project_to_path():
    """Add the implementation/ folder to PYTHONPATH."""
    here = Path(__file__).resolve()
    impl = here.parent.parent
    sys.path.insert(0, str(impl))


_add_project_to_path()


# ---------- helpers ----------

def _load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _save_npy(path: str | Path, arr: np.ndarray):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def step_meta(args):
    """Step 1: Build COCO task metadata for shared1000."""
    from src.data.task_metadata import build_shared1000_metadata
    out_path = Path(args.output_root) / "shared1000_task_metadata.json"
    build_shared1000_metadata(
        nsd_stim_info_csv=Path(args.nsd_root) / "nsddata/experiments/nsd/nsd_stim_info_merged.csv",
        coco_annotations_dir=Path(args.coco_root) / "annotations",
        output_path=out_path,
    )


def step_stim(args):
    """Step 2: Reconstruct 1000 NSD shared stimuli from COCO + cropBox."""
    from src.data.nsd_fsaverage import NSDFsaverageLoader
    from src.data.nsd_stim_reconstruct import NSDStimulusReconstructor

    loader = NSDFsaverageLoader(args.nsd_root)
    nsd_ids = loader.shared_nsd_ids
    logger.info(f"Reconstructing {len(nsd_ids)} shared1000 stimuli")

    rec = NSDStimulusReconstructor(
        stim_info_csv=Path(args.nsd_root) / "nsddata/experiments/nsd/nsd_stim_info_merged.csv",
        coco_train_dir=Path(args.coco_root) / "train2017",
        coco_val_dir=Path(args.coco_root) / "val2017",
    )
    out_dir = Path(args.output_root) / "stimuli_shared1000"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = rec.save_batch(nsd_ids.tolist(), out_dir=out_dir, ext="png")
    logger.info(f"Saved {len(paths)} stimuli to {out_dir}")
    # Save ordering for reference
    with open(out_dir / "manifest.json", "w") as f:
        json.dump({
            "nsd_ids": nsd_ids.tolist(),
            "paths": [str(p.name) for p in paths],
        }, f)


def step_extract(args):
    """Step 3: Extract MLLM activations on shared1000 stimuli."""
    from src.models.extract_activations import ActivationExtractor

    models_cfg = _load_yaml(Path(args.config_dir) / "models.yaml")
    model_cfg = models_cfg["models"][args.model]
    if args.model_path_override:
        model_cfg = dict(model_cfg)
        model_cfg["hf_id"] = args.model_path_override
    out_dir = Path(args.output_root) / "activations" / args.model

    manifest_path = Path(args.output_root) / "stimuli_shared1000" / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    image_paths = [
        str(Path(args.output_root) / "stimuli_shared1000" / p)
        for p in manifest["paths"]
    ]
    nsd_ids = manifest["nsd_ids"]

    extractor = ActivationExtractor(model_cfg, device="cuda", dtype="float16")
    extractor.load_model()

    # Build per-image VQA prompt for activation extraction (free-form description)
    prompts = ["Describe this image in detail."] * len(image_paths)
    extractor.extract_all(
        images=image_paths,
        texts=prompts,
        output_dir=str(out_dir),
        pca_dim=args.pca_dim if args.pca_dim and args.pca_dim > 0 else 0,
    )

    # Save NSD ID ordering — store in both the parent and the model-named subdir
    model_subdir = model_cfg["name"].replace(" ", "_").replace("-", "_")
    actual_out = out_dir / model_subdir
    actual_out.mkdir(parents=True, exist_ok=True)
    for target in (out_dir, actual_out):
        with open(target / "nsd_ids.json", "w") as f:
            json.dump(nsd_ids, f)
    logger.info(f"Extracted activations to {actual_out}")


def step_brain(args):
    """Step 4: Load NSD betas, extract shared1000 per-ROI voxel patterns."""
    from src.data.nsd_fsaverage import (
        NSDFsaverageLoader, NEUROLENS_ROIS, get_roi_voxels,
    )

    loader = NSDFsaverageLoader(args.nsd_root)
    subjects = args.subjects or [1]
    rois = args.rois or list(NEUROLENS_ROIS.keys())
    out_dir = Path(args.output_root) / "brain_data_real"
    out_dir.mkdir(parents=True, exist_ok=True)

    for subj in subjects:
        sessions = loader.available_sessions(subj - 1)
        logger.info(f"subj{subj:02d}: {len(sessions)} sessions available: {sessions}")
        if not sessions:
            logger.warning(f"subj{subj:02d} has no betas available; skipping")
            continue
        for roi in rois:
            try:
                voxels, nsd_ids, ncsnr = get_roi_voxels(
                    loader, subj - 1, roi,
                    max_sessions=args.max_sessions,
                )
            except Exception as e:
                logger.error(f"subj{subj:02d} ROI {roi}: {e}")
                continue
            np.savez(
                out_dir / f"subj{subj:02d}_{roi}.npz",
                voxels=voxels,
                nsd_ids=nsd_ids,
                ncsnr=ncsnr,
            )
            logger.info(
                f"  subj{subj:02d} {roi}: voxels={voxels.shape}, NC mean={ncsnr.mean():.3f}"
            )


def step_encode(args):
    """Step 5: Fit encoding model for each (layer, ROI) per subject."""
    from src.analysis.neurolens_encoding import (
        run_layer_x_roi_encoding, fit_encoding_model_cv,
    )
    from src.data.nsd_fsaverage import NSDFsaverageLoader, NEUROLENS_ROIS

    models_cfg = _load_yaml(Path(args.config_dir) / "models.yaml")
    model_cfg = models_cfg["models"][args.model]
    n_layers = model_cfg["n_layers"]

    # The activation extractor writes to {output_root}/activations/{args.model}/{model_name_underscored}/
    base_act_dir = Path(args.output_root) / "activations" / args.model
    model_subdir = model_cfg["name"].replace(" ", "_").replace("-", "_")
    activations_dir = base_act_dir / model_subdir
    if not activations_dir.exists():
        activations_dir = base_act_dir  # fallback if extractor wrote directly
    nsd_ids_file = activations_dir / "nsd_ids.json"
    if not nsd_ids_file.exists():
        # Look in parent
        nsd_ids_file = base_act_dir / "nsd_ids.json"
    with open(nsd_ids_file) as f:
        activation_nsd_ids = np.array(json.load(f))

    # Load all layer activations. Prefer raw (pre-PCA) if available; the encoding
    # model does its own PCA internally with a sensible component count.
    token_type = args.token_type
    activations_per_layer = {}
    for L in range(n_layers):
        raw_path = activations_dir / f"layer_{L}_{token_type}_raw.npy"
        pca_path = activations_dir / f"layer_{L}_{token_type}.npy"
        path = raw_path if raw_path.exists() else pca_path
        if not path.exists():
            logger.warning(f"Missing layer {L} activation file: {path}")
            continue
        arr = np.load(path).astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] == 0:
            logger.warning(
                f"Layer {L} has degenerate shape {arr.shape}; trying raw fallback"
            )
            if raw_path.exists() and raw_path != path:
                arr = np.load(raw_path).astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] == 0:
            logger.error(f"Layer {L} still degenerate: {arr.shape}; skipping")
            continue
        activations_per_layer[L] = arr
    logger.info(
        f"Loaded {len(activations_per_layer)} layers of activations; "
        f"feature dim = {next(iter(activations_per_layer.values())).shape[1]}"
    )

    loader = NSDFsaverageLoader(args.nsd_root)
    subjects = args.subjects or [1]
    rois = args.rois or list(NEUROLENS_ROIS.keys())
    out_root = Path(args.output_root) / "encoding_real"
    out_root.mkdir(parents=True, exist_ok=True)

    summary_matrices: dict[int, dict] = {}
    for subj in subjects:
        logger.info(f"=== subj{subj:02d} encoding ===")
        result = run_layer_x_roi_encoding(
            activations_per_layer=activations_per_layer,
            nsd_loader=loader,
            subject_idx=subj - 1,
            roi_names=rois,
            activation_nsd_ids=activation_nsd_ids,
            max_sessions=args.max_sessions,
            n_folds=args.n_folds,
            use_pca=args.use_pca,
            n_components=args.pca_dim or 512,
            output_dir=out_root / f"subj{subj:02d}",
        )
        summary_matrices[subj] = result

    # Save aggregate (mean across subjects)
    if summary_matrices:
        mats = [r["encoding_matrix"] for r in summary_matrices.values()]
        norms = [r["encoding_normalized"] for r in summary_matrices.values()]
        ncs = [r["noise_ceiling_per_roi"] for r in summary_matrices.values()]
        np.savez(
            out_root / "encoding_aggregate.npz",
            encoding_matrix_mean=np.mean(mats, axis=0),
            encoding_normalized_mean=np.mean(norms, axis=0),
            noise_ceiling_per_roi_mean=np.mean(ncs, axis=0),
            roi_names=np.array(rois),
            layer_indices=np.array(list(activations_per_layer.keys())),
            subjects=np.array(subjects),
        )
        logger.info(f"Saved encoding aggregate to {out_root}/encoding_aggregate.npz")


def step_cross_modal(args):
    """Step (new): Cross-modal alignment analysis.

    For each (layer L, ROI):
        - Fit Ridge encoding model using ONLY vision tokens (`_vis_raw.npy`).
        - Fit Ridge encoding model using ONLY text tokens (`_txt_raw.npy`).
        - Fit Ridge encoding model using [vis | txt] concatenation.
    Reports per-ROI per-layer vis-only r, txt-only r, and the variance-
    partitioning interaction term (R²_VT - R²_V - R²_T).

    Reuses existing brain voxels from step_brain output and existing
    raw activation files from step_extract.
    """
    from src.analysis.cross_modal import run_cross_modal_encoding

    models_cfg = _load_yaml(Path(args.config_dir) / "models.yaml")
    model_cfg = models_cfg["models"][args.model]
    n_layers = model_cfg["n_layers"]

    # Locate activation files (same layout as step_encode)
    base_act_dir = Path(args.output_root) / "activations" / args.model
    model_subdir = model_cfg["name"].replace(" ", "_").replace("-", "_")
    activations_dir = base_act_dir / model_subdir
    if not activations_dir.exists():
        activations_dir = base_act_dir
    nsd_ids_file = activations_dir / "nsd_ids.json"
    if not nsd_ids_file.exists():
        nsd_ids_file = base_act_dir / "nsd_ids.json"
    with open(nsd_ids_file) as f:
        activation_nsd_ids = np.array(json.load(f))

    # Load per-layer vis and txt activations
    def _load_tokens(token_type: str) -> dict[int, np.ndarray]:
        out: dict[int, np.ndarray] = {}
        for L in range(n_layers):
            for suffix in (f"layer_{L}_{token_type}_raw.npy",
                           f"layer_{L}_{token_type}.npy"):
                p = activations_dir / suffix
                if p.exists():
                    arr = np.load(p).astype(np.float32)
                    if arr.ndim == 2 and arr.shape[1] > 0:
                        out[L] = arr
                        break
        return out

    activations_vis = _load_tokens("vis")
    activations_txt = _load_tokens("txt")
    logger.info(
        f"Loaded {len(activations_vis)} vis layers, "
        f"{len(activations_txt)} txt layers"
    )
    if not activations_vis or not activations_txt:
        raise RuntimeError("Missing vis/txt activations — run step extract first.")

    # Load brain voxels per ROI directly from NSD (uses hemisphere fallback to
    # handle the LH-only-during-download case).
    from src.data.nsd_fsaverage import (
        NSDFsaverageLoader, NEUROLENS_ROIS, get_roi_voxels,
    )

    loader = NSDFsaverageLoader(args.nsd_root)
    subj = args.subjects[0] if args.subjects else 1
    rois_to_use = args.rois or list(NEUROLENS_ROIS.keys())
    brain_per_roi: dict[str, dict] = {}
    for roi in rois_to_use:
        try:
            voxels, ids, ncsnr = get_roi_voxels(
                loader, subj - 1, roi, max_sessions=args.max_sessions,
            )
        except Exception as e:
            logger.warning(f"Skipping ROI {roi}: {e}")
            continue
        if voxels.shape[1] == 0:
            continue
        brain_per_roi[roi] = {
            "voxels": voxels,
            "nsd_ids": ids,
            "ncsnr": ncsnr,
        }
    if not brain_per_roi:
        raise RuntimeError("No ROIs loaded successfully — check NSD data")
    logger.info(f"Loaded {len(brain_per_roi)} ROIs for subj{subj:02d}")

    out_root = Path(args.output_root) / "cross_modal_real"
    run_cross_modal_encoding(
        activations_vis=activations_vis,
        activations_txt=activations_txt,
        brain_per_roi=brain_per_roi,
        activation_nsd_ids=activation_nsd_ids,
        n_folds=args.n_folds,
        n_components=args.pca_dim or 512,
        output_dir=out_root,
    )


def step_sweep(args):
    """Step 6: Single-layer ablation × 3 tasks."""
    from src.data.task_metadata import load_shared1000_metadata, make_task_batches
    from src.data.nsd_fsaverage import NSDFsaverageLoader
    from src.models.extract_activations import ActivationExtractor
    from src.patching.activation_patching import ActivationPatcher
    from src.patching.single_layer_sweep import (
        run_single_layer_sweep, evaluate_object_recognition,
        evaluate_attribute_binding, evaluate_caption_fluency,
        PerplexityScorer, triple_dissociation_from_sweeps,
    )
    from PIL import Image

    models_cfg = _load_yaml(Path(args.config_dir) / "models.yaml")
    model_cfg = models_cfg["models"][args.model]
    if args.model_path_override:
        model_cfg = dict(model_cfg)
        model_cfg["hf_id"] = args.model_path_override
    raw_stages = model_cfg["stages"]
    if isinstance(raw_stages, dict):
        stage_ranges = {
            name: (int(v[0]), int(v[1])) for name, v in raw_stages.items()
        }
    else:
        stage_ranges = {
            f"stage_{i+1}": (int(s[0]), int(s[1]))
            for i, s in enumerate(raw_stages)
        }

    metadata = load_shared1000_metadata(
        Path(args.output_root) / "shared1000_task_metadata.json"
    )

    # Load reconstructed stimuli
    manifest_path = Path(args.output_root) / "stimuli_shared1000" / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    images_dict = {}
    for nid, fname in zip(manifest["nsd_ids"], manifest["paths"]):
        img = Image.open(Path(args.output_root) / "stimuli_shared1000" / fname).convert("RGB")
        images_dict[int(nid)] = img

    # Subsample stimuli for the sweep (full 1000 × 32 layers × 3 tasks is too expensive)
    rng = np.random.default_rng(42)
    sample_nsd_ids = rng.choice(manifest["nsd_ids"], size=args.sweep_n_stim, replace=False)

    tasks = make_task_batches(metadata, sample_nsd_ids, images_dict)

    # Set up extractor / patcher
    extractor = ActivationExtractor(model_cfg, device="cuda", dtype="float16")
    extractor.load_model()
    patcher = ActivationPatcher(
        extractor.model, extractor.processor, model_cfg, device="cuda"
    )
    if getattr(extractor, "_is_manual_llava", False):
        patcher.set_vision_components(
            extractor._vision_tower,
            extractor._image_processor,
            extractor._mm_projector,
        )

    # Compute mean activations needed for mean ablation
    sample_for_means = sample_nsd_ids[: args.mean_n_samples]
    mean_imgs = [images_dict[int(n)] for n in sample_for_means]
    mean_prompts = ["Describe this image."] * len(mean_imgs)
    patcher.compute_mean_activations(mean_imgs, mean_prompts, n_samples=args.mean_n_samples)

    ppl_scorer = None
    if args.run_caption_task:
        ppl_scorer = PerplexityScorer(
            model_name=args.perplexity_model,
            device="cuda",
        )

    out_dir = Path(args.output_root) / "sweep_real"
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep_results = {}

    # Task 1: object recognition
    sweep_results["object"] = run_single_layer_sweep(
        patcher=patcher,
        images=tasks["object"]["images"],
        prompts=tasks["object"]["prompts"],
        ground_truths=tasks["object"]["ground_truths"],
        task_name="object",
        evaluator=evaluate_object_recognition,
        layers_to_sweep=args.sweep_layers,
        method=args.patch_method,
        max_new_tokens=10,
    )

    # Task 2: category MC (binding-like)
    sweep_results["category_mc"] = run_single_layer_sweep(
        patcher=patcher,
        images=tasks["category_mc"]["images"],
        prompts=tasks["category_mc"]["prompts"],
        ground_truths=tasks["category_mc"]["ground_truths"],
        task_name="category_mc",
        evaluator=evaluate_attribute_binding,
        layers_to_sweep=args.sweep_layers,
        method=args.patch_method,
        max_new_tokens=10,
        extra_eval_args={
            "option_lists": tasks["category_mc"]["option_lists"],
            "correct_indices": tasks["category_mc"]["correct_indices"],
        },
    )

    # Task 3: caption fluency
    if args.run_caption_task and ppl_scorer is not None:
        def caption_evaluator(predictions):
            return evaluate_caption_fluency(predictions, ppl_scorer)
        sweep_results["caption"] = run_single_layer_sweep(
            patcher=patcher,
            images=tasks["caption"]["images"],
            prompts=tasks["caption"]["prompts"],
            ground_truths=tasks["caption"]["ground_truths"],
            task_name="caption",
            evaluator=caption_evaluator,
            layers_to_sweep=args.sweep_layers,
            method=args.patch_method,
            max_new_tokens=30,
        )

    # Save raw sweeps
    for tname, res in sweep_results.items():
        np.savez(
            out_dir / f"sweep_{tname}.npz",
            layer_indices=res.layer_indices,
            baseline_score=res.baseline_score,
            patched_scores=res.patched_scores,
            drops=res.drops,
            metric=np.array(res.metric_name),
        )
        with open(out_dir / f"sweep_{tname}_outputs.json", "w") as f:
            json.dump({
                "baseline": res.per_sample_baseline,
                "patched_per_layer": {str(k): v for k, v in res.per_sample_patched.items()},
            }, f, indent=2)

    # Aggregate into dissociation matrix
    dissoc = triple_dissociation_from_sweeps(sweep_results, stage_ranges)
    np.savez(
        out_dir / "dissociation_matrix.npz",
        matrix=dissoc["matrix"],
        stage_names=np.array(dissoc["stage_names"]),
        task_names=np.array(dissoc["task_names"]),
        peak_layer_per_task=dissoc["peak_layer_per_task"],
        diagonal_drops=dissoc["diagonal_drops"],
        dissociation_ratio=dissoc["dissociation_ratio"],
    )
    logger.info(
        f"Triple dissociation matrix shape={dissoc['matrix'].shape}, "
        f"peak layers={dissoc['peak_layer_per_task']}, "
        f"dissoc ratio={dissoc['dissociation_ratio']}"
    )


def step_visualize(args):
    """Step 7: Plot encoding heatmap + dissociation matrix."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(args.output_root) / "figures_real"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Encoding heatmap (per-subject + aggregate)
    enc_root = Path(args.output_root) / "encoding_real"
    agg_path = enc_root / "encoding_aggregate.npz"
    if agg_path.exists():
        agg = np.load(agg_path, allow_pickle=True)
        mat = agg["encoding_normalized_mean"]
        rois = list(agg["roi_names"])
        layers = list(agg["layer_indices"])
        fig, ax = plt.subplots(figsize=(max(6, len(rois) * 0.6), max(5, len(layers) * 0.18)))
        im = ax.imshow(mat, aspect="auto", cmap="viridis", origin="lower",
                       vmin=0, vmax=max(0.5, float(mat.max())))
        ax.set_xticks(range(len(rois)))
        ax.set_xticklabels(rois, rotation=45, ha="right")
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers)
        ax.set_xlabel("Brain ROI")
        ax.set_ylabel("Layer")
        ax.set_title(f"NeuroLens encoding model (NC-normalized r)\n{args.model}")
        fig.colorbar(im, ax=ax, label="Pearson r / NC")
        plt.tight_layout()
        plt.savefig(out_dir / "encoding_heatmap.png", dpi=180)
        plt.close(fig)
        logger.info(f"Saved encoding heatmap to {out_dir / 'encoding_heatmap.png'}")

    # Dissociation matrix
    sweep_root = Path(args.output_root) / "sweep_real"
    dis_path = sweep_root / "dissociation_matrix.npz"
    if dis_path.exists():
        d = np.load(dis_path, allow_pickle=True)
        m = d["matrix"]
        stages = list(d["stage_names"])
        tasks = list(d["task_names"])
        fig, ax = plt.subplots(figsize=(2 + len(tasks), 2 + len(stages)))
        im = ax.imshow(m, aspect="auto", cmap="coolwarm")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center",
                        color="black", fontsize=10)
        ax.set_xticks(range(len(tasks)))
        ax.set_xticklabels(tasks, rotation=15)
        ax.set_yticks(range(len(stages)))
        ax.set_yticklabels(stages)
        ax.set_xlabel("Task")
        ax.set_ylabel("Patched stage")
        ax.set_title("Triple dissociation (single-layer sweep, stage means)")
        fig.colorbar(im, ax=ax, label="Score drop (baseline - patched)")
        plt.tight_layout()
        plt.savefig(out_dir / "dissociation_matrix.png", dpi=180)
        plt.close(fig)
        logger.info(f"Saved dissociation matrix to {out_dir / 'dissociation_matrix.png'}")

    # Cross-modal alignment maps + interaction
    cm_path = Path(args.output_root) / "cross_modal_real" / "cross_modal.npz"
    if cm_path.exists():
        cm = np.load(cm_path, allow_pickle=True)
        rois = list(cm["roi_names"])
        layers = list(cm["layer_indices"])
        stages = list(cm["roi_stages"])
        # Sort ROIs by stage (visual → fusion → language) for cleaner plots
        stage_order = {"visual": 0, "fusion": 1, "language": 2, "other": 3}
        order = sorted(range(len(rois)), key=lambda k: (stage_order[stages[k]], rois[k]))
        rois_s = [rois[k] for k in order]
        stages_s = [stages[k] for k in order]
        r_vis = cm["r_vis_mean"][:, order]
        r_txt = cm["r_txt_mean"][:, order]
        interaction = cm["interaction"][:, order]

        def _heatmap(mat, title, fname, vmin=None, vmax=None, cmap="viridis"):
            fig, ax = plt.subplots(figsize=(max(8, len(rois_s) * 0.6),
                                            max(5, len(layers) * 0.18)))
            im = ax.imshow(mat, aspect="auto", cmap=cmap, origin="lower",
                           vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(rois_s)))
            ax.set_xticklabels(rois_s, rotation=45, ha="right")
            ax.set_yticks(range(len(layers)))
            ax.set_yticklabels(layers)
            ax.set_xlabel("Brain ROI")
            ax.set_ylabel("LLaVA layer")
            ax.set_title(title)
            # Stage boundary lines
            for s in ("visual", "fusion"):
                if s in stages_s:
                    idx = max(i for i, x in enumerate(stages_s) if x == s) + 0.5
                    ax.axvline(idx, color="white", linewidth=2.0, alpha=0.6)
                    ax.axvline(idx, color="black", linewidth=0.7, alpha=0.6)
            fig.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(out_dir / fname, dpi=180)
            plt.close(fig)

        # Compute symmetric vmax for r-style heatmaps
        r_max = float(max(np.abs(r_vis).max(), np.abs(r_txt).max(), 0.05))
        _heatmap(r_vis, "Vision tokens → ROI (mean Pearson r)",
                 "cm_vis_to_roi.png", vmin=-r_max * 0.3, vmax=r_max)
        _heatmap(r_txt, "Text tokens → ROI (mean Pearson r)",
                 "cm_txt_to_roi.png", vmin=-r_max * 0.3, vmax=r_max)

        inter_max = float(max(np.abs(interaction).max(), 1e-3))
        _heatmap(interaction, "Variance-partitioning interaction (R²_VT − R²_V − R²_T)",
                 "cm_interaction.png",
                 vmin=-inter_max, vmax=inter_max, cmap="RdBu_r")

        # Per-layer vis_to_lang and txt_to_vis trajectories (the key cross-modal flow)
        lang_idx = [i for i, s in enumerate(stages_s) if s == "language"]
        vis_idx = [i for i, s in enumerate(stages_s) if s == "visual"]
        if lang_idx and vis_idx:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(layers, r_vis[:, lang_idx].mean(axis=1),
                    label="vis tokens → language ROIs", marker="o", linewidth=2)
            ax.plot(layers, r_txt[:, vis_idx].mean(axis=1),
                    label="txt tokens → visual ROIs", marker="s", linewidth=2)
            ax.plot(layers, r_vis[:, vis_idx].mean(axis=1),
                    label="vis → visual (baseline)", linestyle="--", color="gray")
            ax.plot(layers, r_txt[:, lang_idx].mean(axis=1),
                    label="txt → language (baseline)", linestyle=":", color="gray")
            ax.set_xlabel("LLaVA layer")
            ax.set_ylabel("Mean Pearson r")
            ax.set_title("Cross-modal alignment as a function of layer depth")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / "cm_flow.png", dpi=180)
            plt.close(fig)
            logger.info(f"Saved cross-modal flow plot to {out_dir/'cm_flow.png'}")

        logger.info(f"Saved cross-modal heatmaps to {out_dir}")

    # Per-task layer drop curves
    for task_name in ("object", "category_mc", "caption"):
        sweep_path = sweep_root / f"sweep_{task_name}.npz"
        if not sweep_path.exists():
            continue
        d = np.load(sweep_path)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(d["layer_indices"], d["drops"], marker="o", linewidth=2)
        ax.set_xlabel("Patched layer")
        ax.set_ylabel(f"Score drop ({d['metric'].item()})")
        ax.set_title(f"Single-layer ablation: {task_name}")
        ax.axhline(0, color="gray", linewidth=0.7)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"sweep_{task_name}.png", dpi=180)
        plt.close(fig)
        logger.info(f"Saved sweep plot for {task_name}")


# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description="NeuroLens v2 pipeline")
    p.add_argument("--step", required=True,
                   choices=["meta", "stim", "extract", "brain", "encode",
                            "cross_modal", "sweep", "visualize", "all"])
    p.add_argument("--model", default="llava")
    p.add_argument("--model-path-override", default=None,
                   help="Override hf_id with a local path")
    p.add_argument("--nsd-root", default="/hpc2hdd/home/mzhang630/data/nsd")
    p.add_argument("--coco-root", default="/hpc2hdd/home/yfang870/dataset/coco2017")
    p.add_argument("--output-root", default=os.environ.get("PROJECT_DIR",
                   "/hpc2hdd/home/mzhang630/data/mllm") + "/results_v2")
    p.add_argument("--config-dir", default=str(
        Path(__file__).resolve().parent.parent / "configs"))
    p.add_argument("--subjects", type=int, nargs="*", default=[1])
    p.add_argument("--rois", nargs="*", default=None,
                   help="default: all NEUROLENS_ROIS")
    p.add_argument("--max-sessions", type=int, default=None,
                   help="cap number of NSD sessions per subject (speed)")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--use-pca", action="store_true", default=True)
    p.add_argument("--pca-dim", type=int, default=512)
    p.add_argument("--token-type", default="all",
                   help="activation pooling: 'vis', 'txt', or 'all'")
    p.add_argument("--sweep-n-stim", type=int, default=50,
                   help="number of stimuli per sweep task")
    p.add_argument("--sweep-layers", type=int, nargs="*", default=None,
                   help="layer indices to sweep; default all")
    p.add_argument("--mean-n-samples", type=int, default=20,
                   help="N samples used to compute mean activations for mean ablation")
    p.add_argument("--patch-method", default="mean",
                   choices=["mean", "zero", "noise"])
    p.add_argument("--run-caption-task", action="store_true",
                   help="include perplexity-based caption sweep")
    p.add_argument("--perplexity-model", default="gpt2")
    return p.parse_args()


def main():
    args = parse_args()
    Path(args.output_root).mkdir(parents=True, exist_ok=True)

    steps = [args.step]
    if args.step == "all":
        steps = ["meta", "stim", "extract", "brain", "encode", "sweep", "visualize"]

    dispatch = {
        "meta": step_meta,
        "stim": step_stim,
        "extract": step_extract,
        "brain": step_brain,
        "encode": step_encode,
        "cross_modal": step_cross_modal,
        "sweep": step_sweep,
        "visualize": step_visualize,
    }
    for s in steps:
        logger.info(f"\n========== STEP: {s} ==========")
        dispatch[s](args)
        logger.info(f"========== STEP {s} done ==========\n")


if __name__ == "__main__":
    main()
