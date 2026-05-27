"""Single-layer ablation sweep for fine-grained causal validation.

Replaces the stage-level (11-layer) mean ablation that produced uniform
near-100% drops in the v1 NeuroLens experiment. Instead, ablate each layer
(or small window) individually and measure task-specific performance drop.

Pipeline:
  1. For each layer L in {0..n_layers-1}, patch only that layer.
  2. Compute task-specific score (e.g., VQA exact match, perplexity).
  3. Report per-layer drop = baseline_score - patched_score.

Three task types tested in parallel:
  - object: "What is the main object in this image?" — Stage 1 (visual)
  - attribute: "What color is the X?" / Winoground — Stage 2 (binding)
  - caption: free-form description — Stage 3 (language fluency)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class LayerSweepResult:
    """Per-task per-layer drop record."""
    task_name: str
    layer_indices: np.ndarray            # (n_layers,)
    baseline_score: float
    patched_scores: np.ndarray           # (n_layers,)
    drops: np.ndarray                    # (n_layers,) = baseline - patched
    per_sample_baseline: list            # raw model outputs at baseline
    per_sample_patched: dict             # {layer_idx -> list of outputs}
    metric_name: str

    def stage_means(self, stage_ranges: dict[str, tuple[int, int]]) -> dict[str, float]:
        """Average drop within each stage range (inclusive both ends)."""
        out = {}
        for stage, (lo, hi) in stage_ranges.items():
            mask = (self.layer_indices >= lo) & (self.layer_indices <= hi)
            out[stage] = float(self.drops[mask].mean()) if mask.any() else float("nan")
        return out


def _accuracy_exact_match(predictions: list[str], ground_truths: list[str]) -> float:
    """Case-insensitive substring exact match."""
    if not predictions:
        return 0.0
    hits = 0
    for pred, gt in zip(predictions, ground_truths):
        pred_l = pred.strip().lower()
        gt_l = gt.strip().lower()
        # accept if gt appears anywhere in pred (handles "the answer is X" style)
        if gt_l and gt_l in pred_l:
            hits += 1
    return hits / len(predictions)


def _accuracy_options(predictions: list[str], option_lists: list[list[str]],
                      correct_indices: list[int]) -> float:
    """Multiple-choice accuracy: pick the option with highest substring overlap."""
    hits = 0
    for pred, opts, correct_idx in zip(predictions, option_lists, correct_indices):
        p = pred.strip().lower()
        scores = [opt.lower() in p for opt in opts]
        if any(scores):
            chosen = int(np.argmax(scores))
            if chosen == correct_idx:
                hits += 1
    return hits / max(len(predictions), 1)


class PerplexityScorer:
    """Independent GPT-2 perplexity scorer for caption fluency evaluation."""

    def __init__(self, model_name: str = "gpt2", device: str = "cuda"):
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        logger.info(f"Loading {model_name} for perplexity scoring")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(device).eval()
        self.device = device

    @torch.no_grad()
    def score(self, texts: list[str]) -> np.ndarray:
        """Returns (n,) perplexity per text (lower = more fluent)."""
        ppls = []
        for text in texts:
            text = text.strip()
            if not text:
                ppls.append(float("inf"))
                continue
            ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
            if ids.shape[1] < 2:
                ppls.append(float("inf"))
                continue
            with torch.amp.autocast("cuda", enabled=False):
                out = self.model(ids, labels=ids)
            ppls.append(float(torch.exp(out.loss).item()))
        return np.array(ppls)


def evaluate_object_recognition(
    predictions: list[str],
    ground_truths: list[str],
) -> dict:
    """Evaluate VQA-style object recognition.

    Both predictions and ground_truths are short answer strings.
    """
    acc = _accuracy_exact_match(predictions, ground_truths)
    return {"score": acc, "metric": "exact_match_accuracy"}


def evaluate_attribute_binding(
    predictions: list[str],
    ground_truths: list[str],
    option_lists: Optional[list[list[str]]] = None,
    correct_indices: Optional[list[int]] = None,
) -> dict:
    """Evaluate attribute-object binding.

    If option_lists+correct_indices provided, use multiple-choice accuracy
    (closer to Winoground style). Otherwise exact match.
    """
    if option_lists is not None and correct_indices is not None:
        acc = _accuracy_options(predictions, option_lists, correct_indices)
        return {"score": acc, "metric": "winoground_mc_accuracy"}
    acc = _accuracy_exact_match(predictions, ground_truths)
    return {"score": acc, "metric": "exact_match_accuracy"}


def evaluate_caption_fluency(
    predictions: list[str],
    perplexity_scorer: PerplexityScorer,
    max_perplexity: float = 1000.0,
) -> dict:
    """Evaluate caption fluency via independent GPT-2 perplexity.

    Returns 'score' = 1 - normalized_ppl where higher = better fluency.
    """
    ppls = perplexity_scorer.score(predictions)
    # Cap extreme values; transform to bounded score
    ppls_clipped = np.clip(ppls, 1.0, max_perplexity)
    score = 1.0 - (np.log(ppls_clipped) / np.log(max_perplexity))
    return {
        "score": float(np.mean(score)),
        "metric": "1_minus_normalized_log_perplexity",
        "raw_perplexities": ppls.tolist(),
    }


def run_single_layer_sweep(
    patcher,
    images: Sequence[Image.Image | str],
    prompts: Sequence[str],
    ground_truths: Sequence,
    task_name: str,
    evaluator: Callable[[list[str]], dict],
    layers_to_sweep: Optional[list[int]] = None,
    method: str = "mean",
    max_new_tokens: int = 30,
    batch_log_interval: int = 4,
    extra_eval_args: Optional[dict] = None,
) -> LayerSweepResult:
    """Sweep over single-layer ablations and evaluate task performance.

    Args:
        patcher: ActivationPatcher instance (with mean activations precomputed
                 if using mean ablation)
        images: image inputs
        prompts: text prompts (one per image)
        ground_truths: ground truth answers/captions
        task_name: identifier for this task ('object', 'attribute', 'caption')
        evaluator: callable(predictions) -> {'score': float, ...}
        layers_to_sweep: which layer indices to test (default: all)
        method: 'mean', 'zero', or 'noise'
        max_new_tokens: generation length
        batch_log_interval: print progress every N layers
        extra_eval_args: passed to evaluator as kwargs after predictions

    Returns:
        LayerSweepResult
    """
    n_layers = patcher.config.get("n_layers")
    if n_layers is None:
        # Try to infer from the model
        prefix = patcher._get_layer_prefix()
        module = patcher.model
        for attr in prefix.split("."):
            module = getattr(module, attr)
        n_layers = len(module)
    if layers_to_sweep is None:
        layers_to_sweep = list(range(n_layers))

    eval_kwargs = extra_eval_args or {}

    # Step 1: baseline (no patching)
    logger.info(f"[{task_name}] Computing baseline ({len(images)} samples)")
    baseline_outputs = []
    for img, prompt in zip(images, prompts):
        out = patcher.generate(img, prompt, max_new_tokens=max_new_tokens)
        baseline_outputs.append(out)
    baseline_eval = evaluator(baseline_outputs, ground_truths, **eval_kwargs) \
        if "ground_truths" in evaluator.__code__.co_varnames else \
        evaluator(baseline_outputs, **eval_kwargs)
    baseline_score = float(baseline_eval["score"])
    metric_name = baseline_eval.get("metric", "unknown")
    logger.info(f"[{task_name}] baseline {metric_name}={baseline_score:.4f}")

    # Step 2: sweep
    drops = np.zeros(len(layers_to_sweep), dtype=np.float32)
    patched_scores = np.zeros_like(drops)
    per_layer_outputs: dict[int, list[str]] = {}

    for k, L in enumerate(layers_to_sweep):
        patched_outputs = patcher.patch_stage(
            images=images,
            texts=prompts,
            layer_start=L,
            layer_end=L + 1,
            method=method,
            max_new_tokens=max_new_tokens,
        )
        eval_out = evaluator(patched_outputs, ground_truths, **eval_kwargs) \
            if "ground_truths" in evaluator.__code__.co_varnames else \
            evaluator(patched_outputs, **eval_kwargs)
        patched_scores[k] = float(eval_out["score"])
        drops[k] = baseline_score - patched_scores[k]
        per_layer_outputs[L] = patched_outputs

        if k % batch_log_interval == 0 or k == len(layers_to_sweep) - 1:
            logger.info(
                f"[{task_name}] L{L:2d} | patched={patched_scores[k]:.4f} | "
                f"drop={drops[k]:+.4f}"
            )

    return LayerSweepResult(
        task_name=task_name,
        layer_indices=np.array(layers_to_sweep),
        baseline_score=baseline_score,
        patched_scores=patched_scores,
        drops=drops,
        per_sample_baseline=baseline_outputs,
        per_sample_patched=per_layer_outputs,
        metric_name=metric_name,
    )


# ---------- Triple dissociation summary ----------

def triple_dissociation_from_sweeps(
    sweep_results: dict[str, LayerSweepResult],
    stage_ranges: dict[str, tuple[int, int]],
) -> dict:
    """Aggregate single-layer sweeps into a stage x task dissociation matrix.

    Args:
        sweep_results: {task_name -> LayerSweepResult}
        stage_ranges: {stage_name -> (layer_start, layer_end inclusive)}

    Returns:
        dict with stage x task drop matrix and best-layer-per-task info.
    """
    stage_names = list(stage_ranges.keys())
    task_names = list(sweep_results.keys())
    matrix = np.zeros((len(stage_names), len(task_names)), dtype=np.float32)
    peak_layer = np.zeros(len(task_names), dtype=np.int32)
    peak_drop = np.zeros(len(task_names), dtype=np.float32)

    for j, task in enumerate(task_names):
        res = sweep_results[task]
        means = res.stage_means(stage_ranges)
        for i, stage in enumerate(stage_names):
            matrix[i, j] = means[stage]
        # Locate the peak-drop layer
        peak_idx = int(np.argmax(res.drops))
        peak_layer[j] = int(res.layer_indices[peak_idx])
        peak_drop[j] = float(res.drops[peak_idx])

    # Dissociation strength: max diagonal / max off-diagonal per row
    n = min(matrix.shape)
    diag = np.array([matrix[i, i] for i in range(n)])
    off_diag_max = np.zeros(n)
    for i in range(n):
        row = list(matrix[i, :])
        row.pop(i)
        off_diag_max[i] = max(row) if row else 0.0
    dissoc_ratio = diag / np.maximum(off_diag_max, 1e-6)

    return {
        "matrix": matrix,
        "stage_names": stage_names,
        "task_names": task_names,
        "peak_layer_per_task": peak_layer,
        "peak_drop_per_task": peak_drop,
        "diagonal_drops": diag,
        "off_diagonal_max": off_diag_max,
        "dissociation_ratio": dissoc_ratio,
    }


# ---------- Standard NeuroLens dissociation tasks ----------

def build_default_tasks(
    coco_images_and_metadata: list[dict],
) -> dict[str, dict]:
    """Build the three default NeuroLens task batches from COCO data.

    Args:
        coco_images_and_metadata: list of {
            'image': PIL.Image,
            'image_path': str,
            'category': str,        # main object class
            'attribute': str,       # e.g., 'red car' for binding task
            'attribute_answer': str,  # 'red'
            'attribute_options': [...],
            'attribute_correct_idx': int,
            'caption': str,
        }

    Returns:
        {task_name: {images, prompts, ground_truths, [option_lists, correct_indices]}}
    """
    object_task = {
        "images": [d["image"] for d in coco_images_and_metadata],
        "prompts": ["USER: What is the main object in this image?\nASSISTANT:"
                    for _ in coco_images_and_metadata],
        "ground_truths": [d["category"] for d in coco_images_and_metadata],
    }

    attribute_task = {
        "images": [d["image"] for d in coco_images_and_metadata],
        "prompts": [
            f"USER: {d.get('attribute_prompt', 'What color is the main object?')}\nASSISTANT:"
            for d in coco_images_and_metadata
        ],
        "ground_truths": [d["attribute_answer"] for d in coco_images_and_metadata],
        "option_lists": [d.get("attribute_options", []) for d in coco_images_and_metadata],
        "correct_indices": [d.get("attribute_correct_idx", 0)
                            for d in coco_images_and_metadata],
    }

    caption_task = {
        "images": [d["image"] for d in coco_images_and_metadata],
        "prompts": ["USER: Describe this image in one sentence.\nASSISTANT:"
                    for _ in coco_images_and_metadata],
        "ground_truths": [d["caption"] for d in coco_images_and_metadata],
    }

    return {
        "object": object_task,
        "attribute": attribute_task,
        "caption": caption_task,
    }
