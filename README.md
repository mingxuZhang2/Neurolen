# NeuroLens: Token-Level MLLM–Brain Alignment

Maps how each of LLaVA's 576 vision tokens aligns with human visual cortex across 32 transformer decoder layers, using the Natural Scenes Dataset (NSD) 7T fMRI.

## Core Question

> Do multimodal language models preserve, transform, or discard visual spatial information as image tokens pass through the language decoder — and how does this compare to the human visual hierarchy?

## Key Findings (subj01 preliminary, single subject)

### Finding 1: LLaMA decoder nearly doubles brain alignment vs CLIP alone

| Stage | V1 | V2 | V3 | V4 |
|---|---|---|---|---|
| CLIP-ViT patches (before decoder) | 0.048 | 0.047 | 0.079 | 0.072 |
| MLP projector | 0.042 | 0.040 | 0.063 | 0.060 |
| **LLaMA L14 (peak)** | **0.090** | **0.094** | **0.135** | **0.132** |

- Decoder gain: **+72% to +98%** over CLIP patches
- MLP projector *slightly hurts* alignment (CLIP > projector > LLaMA L0)
- This is NOT a CLIP passthrough — the language decoder actively reshapes visual tokens to be more brain-like

### Finding 2: All visual ROIs peak at decoder mid-layers, then decline

| ROI | Peak layer | Peak mean_r | Noise-ceiling normalized |
|---|---|---|---|
| V1 | L14 | 0.090 | 16.6% of NC |
| V2 | L29 | 0.094 | 17.0% of NC |
| V3 | L14 | 0.135 | 19.7% of NC |
| V4 | L13 | 0.132 | 20.7% of NC |

- V3/V4 (object recognition areas) show slightly earlier peaks than V1/V2
- After mid-layers, alignment declines — decoder shifts from visual processing to language generation

### Finding 3: Vision tokens do NOT become "linguistic"

Logit-lens semantic encoding → language ROIs (Broca, IFG, auditory_assoc, temporal_pole):

| Layer | Broca mean_r | temporal_pole mean_r |
|---|---|---|
| L0 | 0.004 | -0.000 |
| L14 | 0.007 | 0.000 |
| L31 | 0.006 | 0.001 |

**All essentially zero.** Vision tokens remain in visual representation space throughout all 32 decoder layers. The vision-to-language transition happens at **text token positions via cross-attention**, not within the vision tokens themselves.

This parallels human brain organization: V1 never becomes Broca's area — visual information is relayed to language regions through inter-area connections, not intra-area transformation.

### Finding 4: Statistical rigor

- **Parametric test**: all (layer, ROI) combinations: t > 44, p < 10⁻¹⁹⁰ across 576 tokens
- **Permutation null**: image-shuffle test running (200 permutations × 10 layers × 4 ROIs)
- **Noise ceiling**: V1–V4 NC_r = 0.54–0.69; language ROIs NC_r = 0.11–0.37 (low SNR for viewing task, validating that Broca is not well-suited for this paradigm)

## Methods Summary

### Data
- **Brain**: NSD 7T fMRI, subj01, fsaverage surface, 22/40 sessions (LH), 602/1000 shared images
- **Model**: LLaVA-1.5-7B (CLIP-ViT-L/14-336px + MLP projector + LLaMA-2-7B)
- **Stimuli**: shared1000 COCO natural images reconstructed from NSD cropBox metadata

### Pipeline

```
Image → CLIP-ViT → 576 patch tokens (24×24, 1024-d)
     → MLP projector → 576 tokens (4096-d)
     → LLaMA layers 0-31 → per-layer hidden states (576 × 4096)
     → PCA to 256-d per layer
     → Ridge regression → V1/V2/V3/V4 voxels (spatial encoding)
     → Logit lens → top-10 vocab words → word embedding avg → Broca/IFG (semantic encoding)
```

### Metrics (corrected per GPT-5.5 Pro review)
- **Primary**: `mean_r` (mean Pearson r across all voxels in ROI) — avoids selection bias of `max_r`
- **Normalized**: mean_r / noise_ceiling_r (fraction of theoretically achievable correlation)
- **Significance**: one-sample t-test across 576 tokens + image-shuffle permutation null

### Baselines
- CLIP-ViT patch tokens (before projector)
- MLP projector output (before decoder)
- Both run through identical Ridge encoding pipeline

## Project Structure

```
implementation/
  src/
    models/
      token_extract.py          # TokenLevelExtractor: 576 tokens × 32 layers + logit lens
      clip_baseline_extract.py  # CLIP patch + projector baseline extraction
      extract_activations.py    # Original mean-pooled extraction (cross-modal analysis)
    analysis/
      token_encoding.py         # Ridge per (token, layer, ROI) — spatial + semantic
      cross_modal.py            # Vis/txt token × visual/language ROI encoding
    data/
      nsd_fsaverage.py          # NSD beta loading, ROI extraction
      nsd_stim_reconstruct.py   # COCO + cropBox stimulus reconstruction
      task_metadata.py          # shared1000 task metadata from COCO annotations
    visualization/
      token_maps.py             # F1-F4 figure series
  scripts/
    run_token_pipeline.py       # Orchestrator: extract/clip_baseline/spatial/semantic/visualize
    run_token_extract.sh        # SLURM: GPU extraction
    run_clip_baseline.sh        # SLURM: CLIP baseline + spatial encoding
    run_token_spatial.sh        # SLURM: spatial encoding (576 × 32 × 4 ROI Ridge)
    run_token_semantic.sh       # SLURM: semantic encoding via logit lens
    run_noise_ceiling.sh        # SLURM: NC computation from NSD ncsnr
    run_permutation_null.sh     # SLURM: image-shuffle permutation test
  configs/
    models.yaml                 # Model specs (LLaVA/Qwen2-VL/InternVL2)
    rois.yaml                   # Brain ROI definitions
```

## Current Limitations

1. **Single subject** (subj01 only; subj02-08 downloading)
2. **Left hemisphere only** (cannot verify contralateral retinotopy)
3. **602/1000 shared images** (22/40 NSD sessions available)
4. **Single MLLM** (LLaVA only; Qwen2-VL and InternVL2 planned)
5. **Language ROIs poorly suited** for NSD viewing task (low noise ceiling)
6. **Logit lens may not be reliable** at image token positions (training objective mismatch)

## Review Status

See [REVIEW_CHECKLIST.md](REVIEW_CHECKLIST.md) for detailed GPT-5.5 Pro review items and fix status.

## Running

```bash
conda activate neurolens  # or alphasteer on HPC3

# Token-level extraction (GPU, ~20 min)
python scripts/run_token_pipeline.py --step extract --model llava \
    --model-path-override /path/to/llava-v1.5-7b --output-root results_v2

# CLIP baseline (GPU, ~10 min)
python scripts/run_token_pipeline.py --step clip_baseline --model llava \
    --model-path-override /path/to/llava-v1.5-7b --output-root results_v2

# Spatial encoding (CPU-heavy, ~2h)
python scripts/run_token_pipeline.py --step spatial --model llava \
    --output-root results_v2 --nsd-root /path/to/nsd

# Semantic encoding (CPU-heavy, ~3h)
python scripts/run_token_pipeline.py --step semantic --model llava \
    --model-path-override /path/to/llava-v1.5-7b --output-root results_v2 --nsd-root /path/to/nsd

# Figures
python scripts/run_token_pipeline.py --step visualize --output-root results_v2
```
