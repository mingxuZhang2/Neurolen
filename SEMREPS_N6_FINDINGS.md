# SemReps-8K Encoding — N=6 Findings (2026-05-31)

**Core question:** Is the way an MLLM (LLaVA-1.5-7B) processes image+text consistent with how
the human brain does cross-modal matching?

**Design:** Per-subject encoding model (StandardScaler → PCA-512 → RidgeCV), fit on each
subject's *own disjoint* train betas (~4k COCO items), evaluated on the shared 70-item test set.
`within` = MLLM features predict the *matched* modality's brain response (image→SEE, caption→READ);
`cross` = predict the *other* modality. `within ≫ cross` = modality-specific alignment.
Subjects: sub-01/02/03/04/05/07, left hemisphere. Layers [0,4,8,12,14,16,20,24,31].

Run as HPC2 SLURM job 9801694 (`run_semreps_encoding_n6.sh`, partition a128m512u, ~3h).

## Headline

MLLM–brain alignment is **modality-specific and asymmetric**:

- **SEE pathway (strong, clean, 6/6):** MLLM vision tokens align to visual cortex and *only*
  visual cortex. `within ≫ cross` in every visual/association ROI, replicated in all 6 subjects.
- **READ pathway (weak, fuzzy):** alignment concentrates in language/parietal regions, but the
  modality boundary is blurred (vision features partly predict reading too) and the brain's
  reading signal is itself low-SNR. Only language IFG shows a significant modality-specific gap.

## SEE leg — within vs cross (paired t across 6 subjects, peak-within layer)

| ROI              | within | cross  | diff   | t    | p       | sign |
|------------------|--------|--------|--------|------|---------|------|
| early_visual     | +0.164 | −0.002 | +0.166 | 5.28 | 0.0033  | 6/6  |
| ventral_visual   | +0.129 | +0.028 | +0.101 | 4.49 | 0.0065  | 6/6  |
| lateral_temporal | +0.085 | +0.027 | +0.058 | 4.81 | 0.0048  | 6/6  |
| parietal_assoc   | +0.193 | +0.033 | +0.160 | 7.73 | 0.0006  | 6/6  |
| language_ifg     | +0.048 | +0.017 | +0.031 | 1.99 | 0.1027  | 5/6  |

within-vs-0 (one-sample t): early_visual t=10.3 p=0.0001; parietal t=9.2 p=0.0003.

## READ leg — within vs cross (paired t across 6 subjects)

| ROI              | within | cross  | diff   | t    | p      | sign |
|------------------|--------|--------|--------|------|--------|------|
| ventral_visual   | +0.032 | +0.031 | +0.001 | 0.16 | 0.878  | 3/6  |
| lateral_temporal | +0.062 | +0.044 | +0.018 | 1.79 | 0.133  | 4/6  |
| parietal_assoc   | +0.094 | +0.079 | +0.015 | 0.91 | 0.407  | 2/6  |
| language_ifg     | +0.067 | +0.035 | +0.032 | 3.30 | 0.0214 | 5/6  |

Only language IFG shows a significant modality-specific READ component (within-vs-0 t=9.0, p=0.0003).

## Noise-ceiling normalized (model_r / LOO inter-subject ceiling over reliable vertices)

| ROI              | SEE norm | READ norm |
|------------------|----------|-----------|
| early_visual     | 1.14     | 0.39      |
| ventral_visual   | 1.27     | 0.22      |
| lateral_temporal | 1.26     | 0.62      |
| parietal_assoc   | 1.46     | 0.95      |
| language_ifg     | 0.44     | 0.74      |

norm ≈ 1 (or >1) on the SEE leg because the encoder is fit per-subject and captures
subject-specific tuning that the across-subject-mean ceiling washes out — i.e. MLLM vision
features ~saturate the explainable SEE signal. READ ceiling is intrinsically low (means
0.014–0.028 vs SEE 0.04–0.09), so raw READ r looks poor but normalizes to a respectable
0.74–0.95 in language/parietal regions.

## Caveats / not-yet-done

- **This is per-modality representational alignment, not matching-level alignment.** The core
  question ("does MLLM image↔text processing == human cross-modal matching?") is most directly
  answered by testing whether the MLLM's *own* image-caption matching signal tracks the brain's
  matching activity. That experiment is the key next step and is **not yet run**.
- **CLIP-richer-than-text confound.** The SEE>READ asymmetry may partly reflect that CLIP vision
  features are simply richer than short-caption text features, rather than a property of
  cross-modal processing. Control feature richness before attributing the asymmetry to mechanism.
- Magnitude: raw r ≈ 0.16 is modest in absolute terms but normal-to-strong for single-item 7T
  fMRI encoding; normalized it saturates the ceiling.
- Significance above is a random-effects test *across* 6 subjects (large effect sizes, t=5–10).
  A permutation null on the encoding r itself (shuffle stimulus↔response) is **not yet run** for
  N=6 — the RSA phase had perm p<0.01, and a quick re-run via the SLURM wrapper is recommended
  before paper submission.
- Left hemisphere only. Right hemisphere is a cheap add (features are hemisphere-independent;
  only RH beta downloads + an encoding re-run are needed).

**Artifacts:** `encoding_results_multi.json` (per-subject + mean), `noise_ceiling_results.json`,
`encoding_significance_n6.json`. Log: `logs/sr_enc_n6_9801694.log`.
