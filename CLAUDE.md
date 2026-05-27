# NeuroLens

Maps MLLM processing stages to brain functional hierarchy using representational similarity analysis, CKA, encoding models, and causal validation via activation patching.

## Project Structure

```
mllm/
  implementation/
    DESIGN.md                  # Technical design document (read this first)
    environment.yml            # Conda environment: neurolens
    run_experiment.py          # Main entry point (--step extract|brain|rdms|analyze|patch|visualize|all)
    configs/
      models.yaml              # Model HF IDs, layer counts, stage boundaries
      rois.yaml                # Brain ROI definitions (Glasser + Destrieux + NSD)
      experiments.yaml         # Experiment parameters (bootstrap, CV folds, etc.)
    src/
      data/
        nsd_loader.py          # NSD beta loading, ROI extraction, brain RDM computation
        stimulus_align.py      # NSD-COCO alignment, stimulus pair creation
      models/
        extract_activations.py # Hook-based activation extraction (LLaVA, Qwen2-VL, InternVL2)
        model_rdm.py           # Model RDM computation from activation arrays
      analysis/
        rsa_analysis.py        # RSA with bootstrap CI and permutation tests
        cka_analysis.py        # Linear CKA (+ kernel CKA supplementary)
        encoding_models.py     # Ridge regression encoding models with nested CV
        statistical_tests.py   # Bootstrap CI, permutation tests, FDR, noise ceiling
      patching/
        activation_patching.py # Stage-specific mean/zero/noise ablation
        dissociation.py        # Triple dissociation experiment (3 stages x 3 tasks)
        eval_patching.py       # VQA accuracy, binding accuracy, language quality metrics
      visualization/
        heatmaps.py            # Layer x ROI heatmaps, dissociation matrix, metric comparison
        brain_maps.py          # Glass brain plots, surface maps, schematic brain diagrams
    scripts/
      run_extraction.sh        # SLURM: GPU job for activation extraction
      run_analysis.sh          # SLURM: CPU job for RSA/CKA/encoding
      run_patching.sh          # SLURM: GPU job for triple dissociation
      run_all.sh               # Master script: submits all jobs with dependencies
  survey/                      # Literature survey outputs
```

## Key Concepts

### Three Processing Stages
- Stage 1 (Visual Encoding): Early LLM layers (0-10 for 32-layer models)
- Stage 2 (Cross-Modal Fusion): Middle layers (11-22)
- Stage 3 (Linguistic Refinement): Late layers (23-31)

### Brain ROI Groups
- visual_early: V1, V2, V3, V4 (ventral visual stream)
- multimodal_integration: STS, Angular Gyrus, TPJ
- language_network: Broca's area, Wernicke's area (left hemisphere)

### Triple Dissociation (core causal contribution)
Patching each stage selectively impairs its corresponding task type:
- Patch Visual -> impairs object recognition
- Patch Fusion -> impairs attribute-object binding
- Patch Language -> impairs fluent generation

## Running Experiments

```bash
# Setup
conda env create -f implementation/environment.yml
conda activate neurolens

# Full pipeline (set paths first)
export NSD_ROOT=/path/to/nsd
export COCO_ROOT=/path/to/coco

# Individual steps
cd implementation
python run_experiment.py --step extract --model llava --nsd-root $NSD_ROOT --coco-root $COCO_ROOT
python run_experiment.py --step brain --model llava --nsd-root $NSD_ROOT --coco-root $COCO_ROOT
python run_experiment.py --step rdms --model llava
python run_experiment.py --step analyze --model llava
python run_experiment.py --step patch --model llava
python run_experiment.py --step visualize --model llava

# Or run everything
python run_experiment.py --step all --model all --nsd-root $NSD_ROOT --coco-root $COCO_ROOT

# HPC submission
./scripts/run_all.sh $NSD_ROOT $COCO_ROOT
```

## Models
- LLaVA-1.5-7B (primary): liuhaotian/llava-v1.5-7b, 32 layers
- Qwen2-VL-7B: Qwen/Qwen2-VL-7B-Instruct, 28 layers
- InternVL2-8B: OpenGVLab/InternVL2-8B, 32 layers

## Git Workflow
- Branch: `neurolens` (development)
- Merge to `master` after experiments confirm results
