# NeuroLens

Token-level MLLM–brain alignment: how LLaVA's 576 vision tokens map to human visual cortex across 32 decoder layers, using NSD 7T fMRI.

## Current Status (2026-05-27)

**Core finding**: LLaMA decoder reshapes CLIP vision tokens to be ~2× more brain-aligned (mean_r +72-98% over CLIP patches), peaking at mid-layers (L13-14). Vision tokens remain visual throughout — they never become linguistic.

**Completed**: token extraction, spatial encoding (V1-V4), semantic encoding (Broca/IFG), CLIP baseline, noise ceiling, parametric tests.
**Running**: permutation null test.
**Pending**: 8 subjects, multi-MLLM, expanded ROIs, causal ablation.

See [README.md](README.md) for full results and [REVIEW_CHECKLIST.md](REVIEW_CHECKLIST.md) for review status.

## Project Structure

```
implementation/
  src/
    models/
      token_extract.py          # 576 tokens × 32 layers + logit lens (disk-backed memmap)
      clip_baseline_extract.py  # CLIP patch + projector baseline
      extract_activations.py    # Mean-pooled extraction (cross-modal analysis)
    analysis/
      token_encoding.py         # Ridge per (token, layer, ROI) — spatial + streaming semantic
      cross_modal.py            # Vis/txt token × visual/language ROI variance partition
      neurolens_encoding.py     # Layer × ROI encoding with noise ceiling
    data/
      nsd_fsaverage.py          # NSD fsaverage beta loading + ROI masking
      nsd_stim_reconstruct.py   # COCO + cropBox stimulus reconstruction
      task_metadata.py          # shared1000 COCO annotation metadata
    visualization/
      token_maps.py             # F1 retinotopy, F2 logit lens, F3 spatial×semantic, F4 hierarchy
    patching/
      single_layer_sweep.py     # Per-layer ablation × 3 tasks
      activation_patching.py    # Mean/zero/noise ablation
  scripts/
    run_token_pipeline.py       # Main orchestrator (extract/clip_baseline/spatial/semantic/visualize)
    run_pipeline_v2.py          # V2 pipeline (cross-modal, encoding, sweep)
    run_token_*.sh              # SLURM scripts for HPC3
    run_clip_baseline.sh        # CLIP baseline SLURM
    run_noise_ceiling.sh        # Noise ceiling SLURM
    run_permutation_null.sh     # Permutation null SLURM
  configs/
    models.yaml                 # LLaVA / Qwen2-VL / InternVL2 specs
    rois.yaml                   # Brain ROI definitions (19 ROIs)
```

## Key Design Decisions

- **mean_r** (not best_r/max-over-voxels) as primary metric — avoids selection bias
- **Disk-backed memmap** for token extraction — handles 150 GB activation tensor in 80 GB RAM
- **Streaming semantic encoding** — builds per-layer word embeddings lazily to avoid 240 GB materialization
- **CLIP baseline** separation — proves decoder adds brain alignment vs CLIP passthrough

## Running (HPC3)

```bash
# Token extraction (GPU, ~20 min)
sbatch scripts/run_token_extract.sh

# CLIP baseline (GPU, ~10 min)
sbatch scripts/run_clip_baseline.sh

# Spatial encoding (CPU+GPU, ~2h) — depends on extraction
sbatch --dependency=afterok:<extract_job> scripts/run_token_spatial.sh

# Semantic encoding (CPU+GPU, ~3h) — depends on extraction
sbatch --dependency=afterok:<extract_job> scripts/run_token_semantic.sh

# Statistical tests
sbatch scripts/run_noise_ceiling.sh
sbatch scripts/run_permutation_null.sh

# Figures (after spatial+semantic done)
sbatch --dependency=afterok:<spatial>:<semantic> scripts/run_token_visualize.sh
```

## Models
- LLaVA-1.5-7B (primary): CLIP-ViT-L/14-336px + MLP + LLaMA-2-7B, 32 layers, 576 vision tokens
- Qwen2-VL-7B (planned): SigLIP + Qwen2, 28 layers
- InternVL2-8B (planned): InternViT + InternLM2, 32 layers

## Git Workflow
- `main`: clean codebase on GitHub
- `review/gpt-pro-fixes`: current working branch with review checklist
- `neurolens`: local dev branch (contains git history with large files, not pushed)
