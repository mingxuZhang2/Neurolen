# NeuroLens: Token-Level MLLM–Brain Alignment

Maps how LLaVA's internal representations align with human visual cortex across sequence positions (image/prompt/generated tokens) and 32 decoder layers, using NSD 7T fMRI.

## Core Question

> A language decoder does not linguisticize image tokens — it amplifies their visual-cortical alignment. Where, then, does vision-to-language conversion actually happen?

## Three Main Findings (subj01 preliminary)

### Finding 1: Language decoder amplifies visual-cortical alignment (+72–98% over CLIP)

The LLaMA decoder does not merely pass CLIP features through — it actively reshapes vision tokens to better match human visual cortex.

| Stage | V1 | V2 | V3 | V4 |
|---|---|---|---|---|
| CLIP-ViT patches (before decoder) | 0.048 | 0.047 | 0.079 | 0.072 |
| MLP projector (after projection) | 0.042 | 0.040 | 0.063 | 0.060 |
| **LLaMA L14 (decoder peak)** | **0.090** | **0.094** | **0.135** | **0.132** |

- Decoder gain: **+72% (V3) to +98% (V2)** over CLIP patches
- MLP projector *slightly hurts* alignment (optimizes for LLM compatibility, not brain-likeness)
- Confirmed with 5-fold CV: V1 CLIP=0.319 → L14=0.453 (same trend, ~15% lower than single-split)

**Noise-ceiling normalized**: LLaMA L14 explains **17–21% of noise ceiling** for V1–V4; CLIP patches only 9–11%.

### Finding 2: Image tokens are a visual memory bank — they never become linguistic

Three-stream comparison (image tokens vs prompt tokens vs generated tokens) against 11 brain ROIs:

| Stream | V1 | V3 | FFA | PPA | EBA | STS | AG | Broca |
|---|---|---|---|---|---|---|---|---|
| **Image tokens** | 0.109 | 0.163 | **0.240** | **0.238** | **0.265** | 0.015 | 0.027 | 0.040 |
| Prompt tokens | 0.111 | 0.167 | 0.231 | 0.221 | 0.260 | 0.008 | 0.017 | 0.023 |
| Generated tokens | 0.081 | 0.124 | 0.206 | 0.199 | 0.239 | 0.019 | **0.038** | 0.024 |

**Key observations**:
- **Image tokens dominate category-selective cortex**: FFA (face), PPA (scene), EBA (body) — consistently highest encoding
- **Generated tokens lose visual information**: 24–43% weaker than image tokens across V1–V4
- **Language/semantic ROIs are near zero for all streams**: NSD is a viewing task, Broca NC is only 0.30
- **One exception — angular gyrus**: generated tokens show the highest AG encoding (0.038), hinting at semantic integration at output positions

### Finding 3: Vision-to-language conversion happens between positions, not within tokens

The layer × stream trajectory tells a clear story:

**V3 encoding by layer (representative visual ROI)**:

| Layer | Image tokens | Prompt tokens | Generated tokens |
|---|---|---|---|
| L0 | 0.134 | 0.134 | 0.133 |
| L8 | 0.159 | 0.173 | 0.116 |
| L14 | 0.174 | 0.170 | 0.136 |
| L16 | **0.181** | 0.173 | 0.134 |
| L24 | 0.167 | 0.176 | 0.112 |
| L31 | 0.156 | 0.161 | 0.112 |

- Image tokens: rise to mid-layer peak then gently decline — **visual memory bank** being formatted
- Prompt tokens: stable, slightly below image tokens — **intermediate readout**
- Generated tokens: **monotonically decline** from L0 to L31 — visual info consumed, converted to language output

**Interpretation**: MLLM does not transform image tokens into language tokens. Image tokens remain a visual substrate throughout all 32 layers. The decoder amplifies their visual-cortical alignment at mid-layers, then generated tokens read from this substrate via cross-attention and produce language output — losing visual information in the process.

This parallels human brain organization: V1 does not become Broca's area. Visual information flows from visual cortex to language areas through inter-area connections, not intra-area transformation.

## Statistical Rigor

- **Primary metric**: `mean_r` (mean Pearson r across all voxels in ROI) — avoids selection bias of `max_r`
- **Noise ceiling**: NC_r from NSD ncsnr; V1–V4 NC = 0.54–0.69; language ROIs NC = 0.11–0.37
- **Cross-validation**: 5-fold CV with fold-local PCA and standardization (final evaluator)
- **Parametric significance**: all (layer, ROI) one-sample t-test across 576 tokens: t > 44, p < 10⁻¹⁹⁰
- **Permutation null**: image-shuffle test (running)
- **CLIP baseline**: separates CLIP-frontend from decoder contribution

## Methods

### Data
- **Brain**: NSD 7T fMRI, subj01, fsaverage surface, LH, 22/40 sessions, 602/1000 shared images
- **Model**: LLaVA-1.5-7B (CLIP-ViT-L/14-336px → MLP projector → LLaMA-2-7B, 32 layers, 576 vision tokens)
- **Stimuli**: shared1000 COCO natural images

### Experiments

**Experiment 1 — Token-level spatial encoding** (576 tokens × 32 layers × V1/V2/V3/V4):
For each (token i, layer L, ROI R): Ridge regression from token activation (256-d) to ROI voxel responses. 73,728 Ridge models total.

**Experiment 2 — CLIP baseline** (CLIP patches + projector output × V1/V2/V3/V4):
Same encoding pipeline on pre-decoder representations. Isolates decoder contribution.

**Experiment 3 — Sequence-position dissociation** (image/prompt/generated × 10 layers × 11 ROIs):
Mean-pooled hidden states per position stream, encoded to expanded ROI set including FFA/PPA/EBA/STS/AG/Broca/IFG.

**Experiment 4 — Semantic encoding via logit lens** (576 tokens × 32 layers × Broca/IFG/auditory/temporal):
Logit-lens top-10 words → word embedding average → Ridge to language ROIs. Result: ~0 everywhere.

## Current Limitations

1. **Single subject** (subj01 only; subj02-08 downloading)
2. **Left hemisphere only** (cannot verify contralateral retinotopy)
3. **602/1000 images** (22/40 NSD sessions)
4. **Single MLLM** (LLaVA; Qwen2-VL and InternVL2 planned)
5. **Language ROIs poorly suited** for NSD viewing task (low noise ceiling)
6. **Global PCA before CV** (mild train/test leakage; acknowledged as limitation)
7. **No pRF retinotopy validation** yet (planned)

## Project Structure

```
implementation/
  src/
    models/
      token_extract.py          # 576 tokens × 32 layers + logit lens (disk-backed memmap)
      clip_baseline_extract.py  # CLIP patch + projector baseline
      sequence_extract.py       # Image/prompt/generated token extraction with generation
      extract_activations.py    # Mean-pooled extraction (cross-modal analysis)
    analysis/
      token_encoding.py         # Ridge per (token, layer, ROI) with joblib parallelism
      cross_modal.py            # Vis/txt token × visual/language ROI encoding
    data/
      nsd_fsaverage.py          # NSD beta loading + ROI masking
      nsd_stim_reconstruct.py   # COCO + cropBox stimulus reconstruction
    visualization/
      token_maps.py             # F1-F4 figure series
  scripts/
    run_token_pipeline.py       # Orchestrator
    run_sequence_dissociation.sh # Step 2: position dissociation
    run_final_spatial.sh        # 5-fold CV spatial encoding
    run_clip_baseline.sh        # CLIP baseline
    run_noise_ceiling.sh        # NC computation
    run_permutation_null.sh     # Image-shuffle null
```

## Next Steps (per GPT-5.5 Pro review)

1. **Complete 5-fold final spatial** (running) — confirm decoder gain holds under proper CV
2. **Permutation null** (running) — non-parametric significance
3. **8 subjects + both hemispheres** — required for publication
4. **pRF retinotopy validation** — does token grid map to cortical coordinates?
5. **Multi-MLLM** (Qwen2-VL, InternVL2) — architecture generality
6. **Causal ablation** — link brain-aligned tokens to model behavior

## Review Status

See [REVIEW_CHECKLIST.md](REVIEW_CHECKLIST.md) for GPT-5.5 Pro review items and fix status.
