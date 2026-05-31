# NeuroLens: MLLM–Brain Cross-Modal Alignment

Does the way a multimodal LLM (LLaVA-1.5-7B) processes images and text align with how the
human brain performs **cross-modal matching**? We test this with SemReps-8K (6 subjects who
both *view images* and *read captions* of the same COCO scenes), backed by token-level
mechanistic analysis on NSD 7T fMRI.

## Core Question

> When an MLLM relates an image to its caption, does its internal image/text processing map
> onto the brain's own division between *seeing* and *reading* — or onto something else?

---

## Main line — SemReps-8K cross-modal encoding (N=6)

Per-subject encoding model (StandardScaler → PCA-512 → RidgeCV), fit on each subject's *own
disjoint* train betas (~4k COCO items), evaluated on the shared 70-item test set.
`within` = MLLM features predict the *matched* modality's brain response (image→SEE,
caption→READ); `cross` = predict the *other* modality. `within ≫ cross` = modality-specific
alignment. Subjects sub-01/02/03/04/05/07, left hemisphere.

**Headline: MLLM–brain alignment is modality-specific *and asymmetric*.**

- **SEE pathway — strong, clean, replicated 6/6.** MLLM vision tokens align to visual cortex
  and *only* visual cortex; `within ≫ cross` in every visual/association ROI.
- **READ pathway — weak and fuzzy.** Alignment concentrates in language/parietal regions, but
  the modality boundary blurs (vision features partly predict reading too) and the brain's
  reading signal is itself low-SNR. Only language IFG shows a significant modality-specific gap.

### SEE leg — within vs cross (paired t across 6 subjects, peak layer)

| ROI | within | cross | diff | t | p | sign |
|---|---|---|---|---|---|---|
| early_visual | +0.164 | −0.002 | +0.166 | 5.28 | 0.0033 | 6/6 |
| ventral_visual | +0.129 | +0.028 | +0.101 | 4.49 | 0.0065 | 6/6 |
| lateral_temporal | +0.085 | +0.027 | +0.058 | 4.81 | 0.0048 | 6/6 |
| parietal_assoc | +0.193 | +0.033 | +0.160 | 7.73 | 0.0006 | 6/6 |
| language_ifg | +0.048 | +0.017 | +0.031 | 1.99 | 0.1027 | 5/6 |

within-vs-0 one-sample t: early_visual t=10.3 (p=0.0001), parietal t=9.2 (p=0.0003).

### READ leg — only language IFG dissociates

ventral +0.032/+0.031 (p=0.88, no dissociation), parietal +0.094/+0.079 (p=0.41),
lateral_temporal +0.062/+0.044 (p=0.13), **language_ifg +0.067/+0.035 (p=0.021, 5/6)**.

### Noise-ceiling normalized (model_r / LOO inter-subject ceiling)

| ROI | SEE | READ |
|---|---|---|
| early_visual | 1.14 | 0.39 |
| ventral_visual | 1.27 | 0.22 |
| lateral_temporal | 1.26 | 0.62 |
| parietal_assoc | 1.46 | 0.95 |
| language_ifg | 0.44 | 0.74 |

SEE norm ≈ 1 (per-subject encoders capture subject-specific tuning the across-subject ceiling
washes out) → MLLM vision features ~saturate the explainable SEE signal. READ ceiling is
intrinsically low (0.014–0.028 vs SEE 0.04–0.09), so raw READ r looks poor but normalizes to
0.74–0.95 in language/parietal.

### What this does *not* yet show (honest gaps)

- **It is per-modality representational alignment, not matching-level alignment.** The cleanest
  test of the core question is whether the MLLM's *own* image↔caption matching signal tracks the
  brain's matching activity. That experiment is **not yet run** and is the key next step.
- The SEE>READ asymmetry may partly reflect that CLIP vision features are simply richer than
  short-caption text features — a feature-quality confound to rule out, not assume away.
- 70 test items, left hemisphere only, READ low-SNR. Significance above is a random-effects test
  *across* 6 subjects (large effect sizes); a permutation null on the encoding r is still pending.

Artifacts: `SEMREPS_N6_FINDINGS.md`, `encoding_results_multi.json`, `noise_ceiling_results.json`,
`encoding_significance_n6.json`.

---

## Supporting mechanism — NSD token-level analysis (subj01, preliminary)

These results motivate *how* MLLM vision tokens carry visual-cortical structure. They are
**supporting evidence**, not the main claim, and carry important caveats (below).

**S1 — Mid-decoder image-token states predict visual cortex better than CLIP patches**
(+60–88% over CLIP at L14; MLP projector slightly *hurts*). 5-fold CV mean_r V1 0.049→0.082,
V3 0.085→0.136; NC-normalized L14 explains 17–21% of ceiling vs CLIP 9–11%.

**S2 — Image tokens behave as a prompt-invariant visual memory bank.** Image tokens dominate
category-selective cortex (FFA 0.240 / PPA 0.238 / EBA 0.265); generated tokens lose 24–43% of
V1–V4 visual info; cross-prompt cosine = 1.000 for image tokens.

**S3 — Vision→language conversion looks like cross-position routing, not within-token
linguisticization.** Image tokens stay visual across all 32 layers; generated tokens read the
image prefix via causal self-attention and decline monotonically in visual content.

### Caveats that reposition S1–S3 as *supporting* (and that a reviewer will raise)

1. **S1 may be architecture, not language/training.** Our own control (`run_exp_random_llama.sh`)
   randomizes the decoder weights; NSD-phase records indicate the alignment gain largely *survives*
   randomization → the effect is plausibly transformer-depth, not LLaVA-specific learning. This is
   the main reason the NSD decoder-gain story is **not** the headline. (Exact magnitudes to be
   re-verified from the HPC3 logs before any write-up.)
2. **S2's prompt-invariance is partly structural.** Image tokens precede the prompt under a causal
   mask, so they *cannot* attend forward to it — cosine = 1.000 is largely forced by architecture,
   not a surprising discovery. The non-trivial part is that this bank becomes more brain-aligned at
   mid-layers, not that it is prompt-invariant.
3. **S3's "no linguisticization" is probe-limited.** Logit-lens and language-ROI signals are near
   zero, but NSD is a passive *viewing* task with low language-ROI SNR (Broca NC 0.30). The safe
   claim is "no positive evidence of linguisticization under current probes," not a proof of absence.

---

## Statistical rigor & open requirements

- **Primary metric**: `mean_r` over voxels (avoids `max_r` selection bias). SemReps adds
  noise-ceiling normalization (LOO inter-subject reliability).
- **Statistical unit**: subject-level (random effects) / image-level bootstrap — **not** the 576
  tokens (token-level t-tests are sanity checks only; tokens are not independent).
- **Pending P0 (must close before a main-conference submission):**
  - Lock the encoding pipeline: fold-local PCA + scaler + ridge, no leakage (NSD gold-standard rerun).
  - Permutation null / label-shuffle on the encoding r, with FDR correction.
  - Baseline expansion: CLIP layer sweep, DINOv2 / low-level Gabor / random features.
  - Multi-subject + both hemispheres (SemReps is N=6 LH; NSD subj01 only).
- **Pending P1:** matching-level cross-modal analysis (the core test), pRF retinotopy sanity check,
  causal ablation/patching, multi-model replication (Qwen2-VL, InternVL2), caption-embedding probes.

See [REVIEW_CHECKLIST.md](REVIEW_CHECKLIST.md) for the full item-by-item status.

## Methods (data)

- **SemReps-8K** (OpenNeuro ds007272): 6 subjects view images and read captions of COCO scenes;
  fsaverage surface betas; disjoint per-subject train stimuli; shared 70-item test (see+read).
- **NSD** 7T fMRI, subj01, fsaverage LH, shared1000 COCO (supporting analysis).
- **Model**: LLaVA-1.5-7B (CLIP-ViT-L/14-336px → MLP projector → LLaMA-2-7B, 32 layers, 576 tokens).

## Project Structure

```
implementation/
  src/
    models/
      token_extract.py          # 576 tokens × 32 layers + logit lens (disk-backed memmap)
      clip_baseline_extract.py  # CLIP patch + projector baseline
      sequence_extract.py       # Image/prompt/generated token extraction
      extract_activations.py    # Mean-pooled extraction (cross-modal)
    analysis/
      token_encoding.py         # Ridge per (token, layer, ROI)
      cross_modal.py            # Vis/txt token × visual/language ROI encoding
    data/
      nsd_fsaverage.py          # NSD beta loading + ROI masking
  scripts/
    run_semreps_encoding_multi.py / run_semreps_encoding_n6.sh   # SemReps N=6 main line
    run_semreps_noise_ceiling.py                                  # LOO noise-ceiling
    run_semreps_download_betas.sh / _images.py / _extract_s457.sh # SemReps data pipeline
    run_exp_random_llama.sh / run_exp_random_proj.sh              # architecture controls
    run_token_pipeline.py, run_final_spatial.sh, run_clip_baseline.sh  # NSD supporting line
```

## Models
- LLaVA-1.5-7B (primary): CLIP-ViT-L/14-336 + MLP + LLaMA-2-7B, 32 layers, 576 vision tokens
- Qwen2-VL-7B / InternVL2-8B (planned, for architecture generality)
