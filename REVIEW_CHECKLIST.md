# GPT-5.5 Pro Review Checklist

Based on R1 review (2026-05-27) + R2 code review + R3 narrative review.
Mark `[x]` when addressed. Last updated: 2026-05-27 21:45 CST.

---

## Experimental Results Summary

### Completed experiments (7 of 7 core experiments done)
1. **Token-level spatial encoding** (576 tokens × 32 layers × V1/V2/V3/V4) ✅
2. **Token-level semantic encoding** via logit lens (576 × 32 × Broca/IFG/auditory/temporal) ✅
3. **CLIP baseline** (CLIP-ViT patches + MLP projector vs decoder) ✅
4. **Noise ceiling** computation from NSD ncsnr ✅
5. **Parametric significance** (t-test across 576 tokens) ✅
6. **Sequence-position dissociation** (image/prompt/generated × 10 layers × 11 ROIs) ✅
7. **Multi-prompt invariance test** (5 prompts × 3 streams × cosine similarity) ✅
8. **5-fold CV validation** (9 layers × 4 ROIs, confirms single-split trends) ✅
9. Gold-standard raw 4096-d + fold-local PCA encoding 🔄 running (~3h)

### Three core findings

**Finding 1: Language decoder amplifies visual-cortical alignment (+60-88% over CLIP)**

5-fold CV mean_r (primary metric, averaged over 576 tokens):

| Stage | V1 | V2 | V3 | V4 |
|---|---|---|---|---|
| CLIP-ViT patches | 0.049 | 0.047 | 0.085 | 0.075 |
| MLP projector | 0.041 | 0.039 | 0.070 | 0.065 |
| LLaMA L0 | 0.041 | 0.039 | 0.071 | 0.065 |
| **LLaMA L14 (peak)** | **0.082** | **0.088** | **0.136** | **0.132** |
| LLaMA L31 | 0.065 | 0.077 | 0.106 | 0.109 |

Decoder gain over CLIP: **V1 +67%, V2 +88%, V3 +60%, V4 +77%**
NC-normalized: LLaMA L14 explains 17-21% of noise ceiling; CLIP only 9-11%.

**Finding 2: Image tokens are a prompt-invariant visual memory bank**

Cross-prompt cosine similarity (same image, 5 different prompts):

| Stream | Mean cos sim | Min | Max |
|---|---|---|---|
| **Image tokens** | **1.000** | 1.000 | 1.000 |
| Prompt tokens | 0.991 | 0.970 | 1.000 |
| Generated tokens | 0.825 | 0.709 | 0.963 |

Image tokens are **perfectly invariant** to prompt changes. Only output tokens vary.

Sequence-position dissociation (layer-averaged mean_r, 11 ROIs):

| Stream | V1 | V3 | FFA | PPA | EBA | AG | Broca |
|---|---|---|---|---|---|---|---|
| **image tokens** | 0.109 | 0.163 | **0.240** | **0.238** | **0.265** | 0.027 | 0.040 |
| prompt tokens | 0.111 | 0.167 | 0.231 | 0.221 | 0.260 | 0.017 | 0.023 |
| generated tokens | 0.081 | 0.124 | 0.206 | 0.199 | 0.239 | **0.038** | 0.024 |

Image tokens dominate FFA/PPA/EBA. Generated tokens lose 24-43% visual info.

**Finding 3: Vision-to-language conversion is positional routing**

V3 encoding by layer (representative trajectory):

| Layer | Image | Prompt | Generated |
|---|---|---|---|
| L0 | 0.134 | 0.134 | 0.133 |
| L8 | 0.159 | 0.173 | 0.116 |
| L16 | **0.181** | 0.173 | 0.134 |
| L24 | 0.167 | 0.176 | 0.112 |
| L31 | 0.156 | 0.161 | 0.112 |

Image tokens: mid-layer peak then gradual decline (visual memory being formatted).
Generated tokens: monotonic decline (visual info consumed, converted to language).
One exception: AG=0.038 for generated tokens (highest of all streams) — weak semantic integration signal at output positions.

### Noise ceiling

| ROI | NC_r | Interpretation |
|---|---|---|
| V1 | 0.544 | Good SNR for viewing task |
| V2 | 0.548 | Good SNR |
| V3 | 0.688 | Strong SNR |
| V4 | 0.637 | Strong SNR |
| Broca | 0.303 | Low SNR — viewing task not ideal for language ROIs |
| temporal_pole | 0.111 | Very low SNR |

### Revised paper thesis

> A language decoder amplifies human visual-cortical alignment of image tokens without linguisticizing them. Image tokens serve as a prompt-invariant visual memory bank; vision-to-language conversion occurs through sequence-position routing via causal self-attention, not through token-internal transformation.

---

## GPT Pro R1 Concerns — Status

### P0: Metrics & Statistical Rigor

| Concern | Status | Evidence |
|---|---|---|
| Replace best_r with mean_r | ✅ Fixed | All tables now use mean_r as primary. best_r shifted V1 peak from L29→L14 — "reversed hierarchy" invalidated. |
| Noise-ceiling normalization | ✅ Done | L14 explains 17-21% NC; CLIP 9-11%. Language ROIs low NC confirms NSD viewing task limitation. |
| Single-split → 5-fold CV | ✅ Fixed | `_ridge_final_voxelwise` with 5-fold CV. 5-fold results ~15% lower than single-split but same trend. |
| PCA train/test leakage | 🔄 Running | Raw 4096-d extraction done. Gold-standard fold-local PCA encoding (Job 316122) running, ~3h. |
| Permutation null | ❌ Blocked | Job repeatedly hangs at NSD I/O. Parametric test (p<10⁻¹⁹⁰) provides interim evidence. Will retry. |
| Nested CV for voxel selection | ⬜ Planned | Not needed if primary metric is mean_r (no voxel selection). |

### P0: Baselines

| Concern | Status | Evidence |
|---|---|---|
| CLIP patch token baseline | ✅ Done | CLIP=0.049-0.085, projector drops, decoder L14=0.082-0.136. Decoder gain +60-88%. |
| Low-level visual baseline | ⬜ Planned | Gabor/edge/color needed to prove encoding > pixel statistics. |
| CLIP whole-image baseline | ⬜ Planned | CLS token encoding to verify token-level adds value. |

### P1: Interpretation

| Concern | Status | Evidence |
|---|---|---|
| "Token = local patch" framing | ⬜ Acknowledged | Will use "spatially-indexed, globally-contextualized visual tokens." |
| "Reversed hierarchy" claim | ✅ Retracted | mean_r shows all ROIs peak L13-14. No reversal. |
| Language ROI interpretation | ✅ Updated | Expanded to 11 ROIs (FFA/PPA/EBA/STS/AG/Broca/IFG). Broca low NC acknowledged. |
| "cross-attention" wording | ⬜ Will fix | Should be "causal self-attention over image prefix" (decoder-only arch). |

### P1: Logit Lens

| Concern | Status | Evidence |
|---|---|---|
| Logit-lens near zero | ✅ Documented | All semantic mean_r ≈ 0.001-0.008. Reported as negative finding. |
| Caption-embedding probe | ⬜ Planned | Better semantic readout than raw logit lens. |
| Tuned Lens | ⬜ Planned | More stable early-layer readout. |

### P1: Data Completeness

| Concern | Status |
|---|---|
| 8 subjects | ⬜ Downloading |
| Both hemispheres | ⬜ Needs RH betas |
| Full 1000 images | ⬜ Needs all 40 sessions |

## GPT Pro R2 Concerns — Status

| Concern | Status | Evidence |
|---|---|---|
| Code used single-split not 5-fold | ✅ Fixed | `_ridge_final_voxelwise` + `run_final_spatial.sh`. Results consistent. |
| PCA fit before CV split | 🔄 Running | Raw 4096-d extracted. Gold-standard job 316122 with fold-local PCA from raw features. |
| t-test across 576 tokens not valid as main stat | ✅ Acknowledged | Demoted to sanity check. Main stat will be paired stage contrast + bootstrap CI. |
| Permutation null too coarse | ❌ Blocked | NSD I/O hang. Will redesign with smaller token sample. |

## GPT Pro R3 Concerns — Status

| Concern | Status | Evidence |
|---|---|---|
| README PCA contradiction | ⬜ Will fix | Need to unify "fold-local PCA" vs "global PCA leakage" language. |
| 0.048 vs 0.319 metric confusion | ✅ Fixed | 5-fold table now primary. All best_r values labeled explicitly. |
| "cross-attention" → "causal self-attention" | ⬜ Will fix in README |
| "do NOT become linguistic" too strong | ⬜ Will soften | → "no evidence for linguisticization under current probes" |
| "V1 never becomes Broca" too strong | ⬜ Will soften | → "division-of-labor reminiscent of cortical specialization" |
| Prompt tokens ≈ image tokens in V1/V3 | ✅ Noted | Prompt tokens slightly higher in V1/V3 (attend to image prefix). Image tokens higher in FFA/PPA/EBA. |
| AG=0.038 too weak for "language conversion" | ✅ Acknowledged | Described as "weak preliminary signal" not strong conclusion. |
| CLIP baseline needs layer sweep | ⬜ Planned | CLIP multi-layer + DINOv2 + low-level baselines. |
| Need multi-prompt test | ✅ Done | 5 prompts × 3 streams. Image tokens cos_sim=1.000. Prompt=0.991. Generated=0.825. |
| Need caption embedding probe | ⬜ Planned | MPNet/CLIP-text embedding prediction for generated tokens. |
| Need teacher-forced control | ⬜ Planned | Fixed caption prefix → extract at caption token positions. |

---

## Experiment Completion Matrix

| Experiment | R1 | R2 | R3 | Status |
|---|---|---|---|---|
| Token-level spatial encoding (mean_r) | Required | — | — | ✅ |
| CLIP baseline | Required | — | — | ✅ |
| Noise ceiling | Required | — | — | ✅ |
| 5-fold CV | — | Required | — | ✅ |
| Raw 4096-d fold-local PCA | — | Required | — | 🔄 Running |
| Sequence-position dissociation | — | — | Suggested | ✅ |
| Multi-prompt invariance | — | — | Suggested | ✅ |
| Permutation null | Required | — | — | ❌ I/O blocked |
| pRF retinotopy | Required | — | Required | ⬜ Data ready |
| Caption embedding probe | — | — | Required | ⬜ Planned |
| Multi-model (Qwen/InternVL) | — | — | — | ⬜ Planned |
| 8 subjects | Required | — | Required | ⬜ Downloading |
| Causal ablation | — | — | Suggested | ⬜ Planned |

---

## Git Branch Status

- `main`: clean codebase baseline
- `review/gpt-pro-fixes`: all fixes + new experiments (current working branch)
- Commits since R1: mean_r fix, CLIP baseline, noise ceiling, 5-fold CV, token-level extraction with memmap, sequence-position dissociation, multi-prompt, raw 4096-d extraction, joblib parallelization
