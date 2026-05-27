"""Triple dissociation experiment runner.

Implements the neuroscience-style triple dissociation as the core causal
validation. Tests 3 stages x 3 task types to show that each MLLM stage
selectively supports a different cognitive function.

The diagonal pattern in the 3x3 matrix = triple dissociation = strongest
causal evidence.

Task types:
1. Visual Recognition: "What object is this?" -- tests pure visual identification
2. Cross-Modal Binding: "What color is the [object]?" with multi-object scenes
3. Language Generation: open-ended captioning -- tests linguistic fluency

Patched stages:
1. Visual (layers 0-10): should selectively impair visual recognition
2. Fusion (layers 11-22): should selectively impair binding
3. Language (layers 23-31): should selectively impair language generation
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.patching.activation_patching import ActivationPatcher
from src.patching.eval_patching import (
    evaluate_vqa_accuracy,
    evaluate_binding_accuracy,
    evaluate_language_quality,
)

logger = logging.getLogger(__name__)


@dataclass
class DissociationTask:
    """A single task instance for dissociation testing."""
    task_type: str          # "visual_recognition", "cross_modal_binding", "language_generation"
    image_path: str
    prompt: str
    ground_truth: str       # expected answer or reference caption
    metadata: dict = field(default_factory=dict)


@dataclass
class DissociationResult:
    """Result of patching one stage on one task type."""
    stage: str
    task_type: str
    baseline_score: float
    patched_score: float
    performance_drop: float
    n_samples: int
    per_sample_scores: list = field(default_factory=list)


class TripleDissociationExperiment:
    """Runs the full 3x3 triple dissociation experiment."""

    STAGES = ["visual", "fusion", "language"]
    TASK_TYPES = ["visual_recognition", "cross_modal_binding", "language_generation"]

    # Brain lesion analogies for each cell
    LESION_ANALOGIES = {
        ("visual", "visual_recognition"):
            "V1-V4 damage -> cortical blindness / visual agnosia. "
            "Patient cannot identify objects despite intact eyes.",
        ("visual", "cross_modal_binding"):
            "V1-V4 damage -> mild binding impairment. "
            "Some feature extraction loss but binding machinery intact.",
        ("visual", "language_generation"):
            "V1-V4 damage -> language unaffected. "
            "Patient can still speak fluently; only visual input is lost.",
        ("fusion", "visual_recognition"):
            "Parietal/temporal damage -> visual recognition mostly preserved. "
            "Objects still identified but integration with other features impaired.",
        ("fusion", "cross_modal_binding"):
            "TPJ/parietal damage -> Balint's syndrome features. "
            "Patient sees features correctly but cannot bind attributes to objects.",
        ("fusion", "language_generation"):
            "Parietal/temporal damage -> language mostly preserved. "
            "May show some anomia but fluency intact.",
        ("language", "visual_recognition"):
            "Broca's area damage -> visual recognition intact. "
            "Patient understands visual scenes; expressive aphasia only.",
        ("language", "cross_modal_binding"):
            "Broca's area damage -> binding intact. "
            "Patient can match attributes to objects; output language is impaired.",
        ("language", "language_generation"):
            "Broca's area damage -> non-fluent (expressive) aphasia. "
            "Patient cannot produce coherent language despite understanding.",
    }

    def __init__(
        self,
        patcher: ActivationPatcher,
        patch_type: str = "mean",
    ):
        self.patcher = patcher
        self.patch_type = patch_type
        self.results: dict[tuple[str, str], DissociationResult] = {}

    def prepare_tasks(
        self,
        image_dir: str,
        task_file: str,
        n_samples_per_type: int = 200,
    ) -> dict[str, list[DissociationTask]]:
        """Load or generate task instances for each task type.

        Args:
            image_dir: directory containing images
            task_file: JSON file with task definitions
            n_samples_per_type: how many samples per task type

        Returns:
            dict mapping task_type -> list of DissociationTask
        """
        if Path(task_file).exists():
            with open(task_file) as f:
                raw_tasks = json.load(f)
            tasks = {}
            for task_type, items in raw_tasks.items():
                tasks[task_type] = [
                    DissociationTask(
                        task_type=task_type,
                        image_path=item["image_path"],
                        prompt=item["prompt"],
                        ground_truth=item["ground_truth"],
                        metadata=item.get("metadata", {}),
                    )
                    for item in items[:n_samples_per_type]
                ]
            return tasks

        # Generate tasks from images if no task file exists
        return self._generate_default_tasks(image_dir, n_samples_per_type)

    def _generate_default_tasks(
        self,
        image_dir: str,
        n_samples: int,
    ) -> dict[str, list[DissociationTask]]:
        """Generate default task instances from available images."""
        img_dir = Path(image_dir)
        image_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        image_paths = image_paths[:n_samples * 3]  # need enough for all task types

        tasks = {t: [] for t in self.TASK_TYPES}

        # Visual Recognition tasks
        vr_prompts = [
            "What is the main object in this image? Answer with a single word or short phrase.",
            "Identify the primary object visible in this image.",
            "What type of object is shown in the center of this image?",
        ]
        for i, img_path in enumerate(image_paths[:n_samples]):
            tasks["visual_recognition"].append(DissociationTask(
                task_type="visual_recognition",
                image_path=str(img_path),
                prompt=vr_prompts[i % len(vr_prompts)],
                ground_truth="",  # will be filled by baseline run
            ))

        # Cross-Modal Binding tasks
        binding_prompts = [
            "What color is the largest object in this image?",
            "Describe the spatial relationship between the objects in this image.",
            "What is the color and position of each distinct object?",
        ]
        for i, img_path in enumerate(image_paths[n_samples:2 * n_samples]):
            tasks["cross_modal_binding"].append(DissociationTask(
                task_type="cross_modal_binding",
                image_path=str(img_path),
                prompt=binding_prompts[i % len(binding_prompts)],
                ground_truth="",
            ))

        # Language Generation tasks
        gen_prompts = [
            "Describe this image in detail.",
            "Write a comprehensive caption for this image.",
            "Provide a thorough description of what you see in this image.",
        ]
        for i, img_path in enumerate(image_paths[2 * n_samples:3 * n_samples]):
            tasks["language_generation"].append(DissociationTask(
                task_type="language_generation",
                image_path=str(img_path),
                prompt=gen_prompts[i % len(gen_prompts)],
                ground_truth="",
            ))

        return tasks

    def run_baseline(
        self,
        tasks: dict[str, list[DissociationTask]],
    ) -> dict[str, list[str]]:
        """Run model without patching to establish baselines.

        Returns:
            dict mapping task_type -> list of baseline responses
        """
        baseline_responses = {}

        for task_type, task_list in tasks.items():
            logger.info(f"Running baseline for {task_type} ({len(task_list)} samples)")
            responses = []
            for task in tqdm(task_list, desc=f"Baseline {task_type}"):
                result = self.patcher.patch_and_generate(
                    image=task.image_path,
                    text=task.prompt,
                    layer_range=(0, 0),  # dummy range, no actual patching
                    patch_type="zero",   # patching 0 layers = no effect
                    max_new_tokens=128,
                )
                # Actually, we should run without any patching at all
                # Remove hooks and run clean
                self.patcher._remove_hooks()
                clean_result = self._run_clean(task.image_path, task.prompt)
                responses.append(clean_result)

                # Store as ground truth if not provided
                if not task.ground_truth:
                    task.ground_truth = clean_result

            baseline_responses[task_type] = responses
            logger.info(f"Baseline {task_type}: {len(responses)} responses collected")

        return baseline_responses

    def _run_clean(self, image_path: str, text: str) -> str:
        """Run model without any patching."""
        image = Image.open(image_path).convert("RGB")
        inputs = self.patcher._prepare_inputs(image, text)
        with torch.no_grad():
            result = self.patcher._run_generate(inputs, max_new_tokens=128)
        return result["text"]

    def run_patching(
        self,
        tasks: dict[str, list[DissociationTask]],
        baseline_responses: dict[str, list[str]],
    ):
        """Run all 3 stages x 3 task types patching experiments.

        Populates self.results with DissociationResult for each cell.
        """
        for stage in self.STAGES:
            layer_range = self.patcher.get_stage_layer_range(stage)
            logger.info(f"\nPatching stage: {stage} (layers {layer_range[0]}-{layer_range[1]})")

            for task_type in self.TASK_TYPES:
                logger.info(f"  Task type: {task_type}")
                task_list = tasks[task_type]
                baselines = baseline_responses[task_type]

                patched_responses = []
                for task in tqdm(task_list,
                                 desc=f"Patch {stage}/{task_type}"):
                    result = self.patcher.patch_and_generate(
                        image=task.image_path,
                        text=task.prompt,
                        layer_range=layer_range,
                        patch_type=self.patch_type,
                        max_new_tokens=128,
                    )
                    patched_responses.append(result["text"])

                # Evaluate
                baseline_score, patched_score, per_sample = self._evaluate(
                    task_type, task_list, baselines, patched_responses
                )

                drop = baseline_score - patched_score
                self.results[(stage, task_type)] = DissociationResult(
                    stage=stage,
                    task_type=task_type,
                    baseline_score=baseline_score,
                    patched_score=patched_score,
                    performance_drop=drop,
                    n_samples=len(task_list),
                    per_sample_scores=per_sample,
                )

                logger.info(f"  {stage}/{task_type}: baseline={baseline_score:.3f}, "
                            f"patched={patched_score:.3f}, drop={drop:.3f}")

    def _evaluate(
        self,
        task_type: str,
        tasks: list[DissociationTask],
        baselines: list[str],
        patched: list[str],
    ) -> tuple[float, float, list]:
        """Evaluate baseline and patched responses for a task type."""
        ground_truths = [t.ground_truth for t in tasks]

        if task_type == "visual_recognition":
            baseline_score = evaluate_vqa_accuracy(baselines, ground_truths)
            patched_score = evaluate_vqa_accuracy(patched, ground_truths)
            per_sample = [
                {"baseline": b, "patched": p, "gt": gt}
                for b, p, gt in zip(baselines, patched, ground_truths)
            ]
        elif task_type == "cross_modal_binding":
            baseline_score = evaluate_binding_accuracy(baselines, ground_truths)
            patched_score = evaluate_binding_accuracy(patched, ground_truths)
            per_sample = [
                {"baseline": b, "patched": p, "gt": gt}
                for b, p, gt in zip(baselines, patched, ground_truths)
            ]
        elif task_type == "language_generation":
            baseline_quality = evaluate_language_quality(baselines)
            patched_quality = evaluate_language_quality(patched)
            baseline_score = baseline_quality["composite"]
            patched_score = patched_quality["composite"]
            per_sample = [
                {"baseline": b, "patched": p,
                 "baseline_quality": baseline_quality,
                 "patched_quality": patched_quality}
                for b, p in zip(baselines, patched)
            ]
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        return baseline_score, patched_score, per_sample

    def get_dissociation_matrix(self) -> np.ndarray:
        """Extract the 3x3 dissociation matrix (performance drops).

        Rows: patched stages (visual, fusion, language)
        Columns: task types (visual_recognition, cross_modal_binding, language_generation)
        Values: performance drop (baseline - patched), larger = more impairment

        Returns:
            (3, 3) numpy array
        """
        matrix = np.zeros((3, 3))
        for i, stage in enumerate(self.STAGES):
            for j, task_type in enumerate(self.TASK_TYPES):
                key = (stage, task_type)
                if key in self.results:
                    matrix[i, j] = self.results[key].performance_drop
        return matrix

    def compute_dissociation_statistics(self) -> dict:
        """Compute statistics on the dissociation pattern.

        Returns:
            dict with selectivity indices, interaction effects, and
            statistical tests for the triple dissociation.
        """
        from src.analysis.statistical_tests import test_dissociation

        matrix = self.get_dissociation_matrix()
        stats_result = test_dissociation(
            matrix,
            row_labels=self.STAGES,
            col_labels=self.TASK_TYPES,
        )

        # Add brain lesion analogies
        analogies = {}
        for i, stage in enumerate(self.STAGES):
            for j, task_type in enumerate(self.TASK_TYPES):
                key = (stage, task_type)
                analogies[f"{stage}_{task_type}"] = {
                    "drop": float(matrix[i, j]),
                    "is_diagonal": i == j,
                    "brain_analogy": self.LESION_ANALOGIES.get(key, ""),
                    "expected": "BIG DROP" if i == j else "small/no change",
                }

        stats_result["analogies"] = analogies
        stats_result["dissociation_matrix"] = matrix
        return stats_result

    def save_results(self, output_dir: str):
        """Save all results to disk."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Save dissociation matrix
        matrix = self.get_dissociation_matrix()
        np.save(out / "dissociation_matrix.npy", matrix)

        # Save full results
        results_dict = {}
        for (stage, task_type), result in self.results.items():
            results_dict[f"{stage}__{task_type}"] = {
                "stage": result.stage,
                "task_type": result.task_type,
                "baseline_score": result.baseline_score,
                "patched_score": result.patched_score,
                "performance_drop": result.performance_drop,
                "n_samples": result.n_samples,
            }

        with open(out / "dissociation_results.json", 'w') as f:
            json.dump(results_dict, f, indent=2)

        # Save statistics
        stats = self.compute_dissociation_statistics()
        # Convert numpy arrays for JSON serialization
        stats_serializable = {}
        for k, v in stats.items():
            if isinstance(v, np.ndarray):
                stats_serializable[k] = v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                stats_serializable[k] = float(v)
            else:
                stats_serializable[k] = v
        with open(out / "dissociation_statistics.json", 'w') as f:
            json.dump(stats_serializable, f, indent=2, default=str)

        logger.info(f"Saved dissociation results to {out}")


def run_triple_dissociation(
    model_config: dict,
    model,
    processor,
    image_dir: str,
    task_file: str = "",
    n_samples: int = 200,
    patch_type: str = "mean",
    mean_activation_images: list = None,
    mean_activation_texts: list = None,
    output_dir: str = "results/dissociation",
) -> dict:
    """Run the complete triple dissociation experiment.

    Args:
        model_config: model configuration dict
        model: loaded model
        processor: loaded processor/tokenizer
        image_dir: directory with evaluation images
        task_file: JSON file defining tasks (optional)
        n_samples: samples per task type
        patch_type: 'mean' or 'zero'
        mean_activation_images: images for computing mean activations
        mean_activation_texts: texts for computing mean activations
        output_dir: where to save results

    Returns:
        dict with dissociation matrix and statistics
    """
    patcher = ActivationPatcher(model, processor, model_config)

    # Compute mean activations for mean ablation
    if patch_type == "mean" and mean_activation_images is not None:
        patcher.compute_mean_activations(
            mean_activation_images, mean_activation_texts, n_samples=100
        )

    experiment = TripleDissociationExperiment(patcher, patch_type)

    # Prepare tasks
    tasks = experiment.prepare_tasks(image_dir, task_file, n_samples)

    # Run baseline
    baseline_responses = experiment.run_baseline(tasks)

    # Run patching for all 3x3 conditions
    experiment.run_patching(tasks, baseline_responses)

    # Save results
    experiment.save_results(output_dir)

    # Return summary
    return experiment.compute_dissociation_statistics()
