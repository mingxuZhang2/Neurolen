# GPT-5.5 Pro Review Checklist

Based on R1 review (2026-05-27) + R2 code review + R3 narrative review + R4 literature-positioning
review (2026-05-31). Mark `[x]` when addressed. Last updated: 2026-05-31 CST.

> **R4 strategic reframe (current direction):** the project spine is now the **SemReps-8K
> cross-modal encoding (N=6)**, which answers the committed core question and survives the
> architecture controls. The NSD token-level decoder-gain story (Findings 1–3 below) is **demoted
> to supporting mechanism** because (a) our own random-LLaMA control suggests the decoder gain is
> largely *architecture, not training*, and (b) the "prompt-invariant memory bank" is partly a
> causal-mask necessity. See the R4 section near the bottom for the full status.

---

## Experimental Results Summary

### Experiment status

Core pilot experiments complete; publication-critical validation remains in progress.

1. **Token-level spatial encoding** (576 tokens × 32 layers × V1/V2/V3/V4) ✅
2. **Token-level semantic encoding** via logit lens (576 × 32 × Broca/IFG/auditory/temporal) ✅
3. **CLIP baseline** (CLIP-ViT patches + MLP projector vs decoder) ✅
4. **Noise ceiling** computation from NSD ncsnr ✅
5. **Parametric significance** (t-test across 576 tokens, demoted to sanity check) ✅
6. **Sequence-position dissociation** (image/prompt/generated × 10 layers × 11 ROIs) ✅
7. **Multi-prompt invariance test** (5 prompts × 3 streams × cosine similarity) ✅
8. **5-fold CV validation** (9 layers × 4 ROIs, confirms single-split trends) ✅
9. Gold-standard raw 4096-d + fold-local PCA encoding 🔄 rerunning (CLIP/projector confirmed; decoder layers need memory fix)

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

### Revised paper thesis (R4)

> An MLLM's image and text processing map onto the human brain's *seeing/reading* division in a
> **modality-specific but asymmetric** way: vision tokens align tightly and specifically with
> visual cortex (image→SEE strong, image→READ near zero, 6/6 subjects), whereas caption features
> align only weakly and mainly in language IFG. The token-level NSD mechanism (decoder reshaping a
> visual memory bank, cross-position routing) is supporting detail, with the decoder gain flagged
> as plausibly architecture-driven rather than language-specific.

> (superseded R3 thesis: "A language decoder amplifies visual-cortical alignment of image tokens
> without linguisticizing them …" — demoted to supporting mechanism per R4.)

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
| "cross-attention" wording | ✅ Fixed | Changed to "causal self-attention over image prefix" in README. |

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
| README PCA contradiction | ✅ Fixed | Limitation now reads: "Current results use 5-fold CV on pre-PCA'd features. Gold-standard raw 4096-d fold-local PCA rerun in progress." |
| 0.048 vs 0.319 metric confusion | ✅ Fixed | Removed best_r from 5-fold line; now shows mean_r only (0.049→0.082 etc). |
| "cross-attention" → "causal self-attention" | ✅ Fixed | README updated to "causal self-attention over image prefix" |
| "do NOT become linguistic" too strong | ✅ Fixed | → "no evidence under current probes" |
| "V1 never becomes Broca" too strong | ✅ Fixed | → "division-of-labor reminiscent of cortical specialization" |
| Prompt tokens ≈ image tokens in V1/V3 | ✅ Noted | Prompt tokens slightly higher in V1/V3 (attend to image prefix). Image tokens higher in FFA/PPA/EBA. |
| AG=0.038 too weak for "language conversion" | ✅ Acknowledged | Described as "weak preliminary signal" not strong conclusion. |
| CLIP baseline needs layer sweep | ⬜ Planned | CLIP multi-layer + DINOv2 + low-level baselines. |
| Need multi-prompt test | ✅ Done | 5 prompts × 3 streams. Image tokens cos_sim=1.000. Prompt=0.991. Generated=0.825. |
| Need caption embedding probe | ⬜ Planned | MPNet/CLIP-text embedding prediction for generated tokens. |
| Need teacher-forced control | ⬜ Planned | Fixed caption prefix → extract at caption token positions. |

---

## GPT Pro R4 Concerns (literature-positioning, 2026-05-31) — Status

### Strategic / framing

| Concern | Status | Resolution |
|---|---|---|
| Recommended headline "Language Decoders Amplify Visual-Cortical Alignment" | ⚠️ Rejected as headline | Undercut by our own random-LLaMA control (architecture, not training) — reviewer would sink it. Demoted to supporting mechanism. |
| SemReps treated only as "external validation" (Claim D) | ✅ Reframed | SemReps cross-modal is now the **spine** (it is MLLM-specific and on-question); NSD is support. |
| "no linguisticization" too strong | ✅ Softened | README: "no positive evidence under current probes"; NSD language-ROI low SNR noted. |
| "prompt-invariant memory bank" reads as a discovery | ✅ Softened | README S2 caveat: cosine=1.000 is largely forced by the causal mask (image precedes prompt). |
| Over-strong "MLLM three-stage == brain three-stage" analogy | ✅ Avoided | Framed as "computationally resembles," not "proves same mechanism." |

### P0 (must close before main-conference submission) — endorsed

| Item | Status | Note |
|---|---|---|
| Lock encoding pipeline (fold-local PCA/scaler/ridge, no leakage) | 🔄 NSD rerun in progress | SemReps multi already fits PCA/scaler inside per-subject train only. |
| Permutation null on encoding r + FDR | ⬜ Pending | SemReps N=6 currently uses across-subject random-effects t (large effects); add shuffle null. |
| Statistical unit = subject / image bootstrap, not 576 tokens | ✅ Done for SemReps | N=6 significance is paired t across subjects. |
| Baseline expansion (CLIP layer sweep, DINOv2, low-level Gabor, random features) | ⬜ Planned | Applies to whichever leg is headlined. |
| Multi-subject + both hemispheres | 🟡 Partial | SemReps N=6 done (LH); RH is a cheap add (features hemisphere-independent). NSD still subj01. |
| Re-verify random-LLaMA magnitude from HPC3 logs | ⬜ TODO | Prerequisite to deciding if the NSD decoder-gain line is salvageable at all. |

### P1 (turns it into a strong paper) — endorsed

| Item | Status | Note |
|---|---|---|
| **Matching-level cross-modal analysis** (the real test of the core question) | ⬜ Key next step | Does the MLLM's own image↔caption matching signal track the brain's matching activity? Current result is per-modality alignment only. |
| Rule out CLIP-richer-than-text confound for SEE>READ asymmetry | ⬜ Pending | Control feature richness before attributing asymmetry to cross-modal processing. |
| pRF / retinotopy sanity check | ⬜ Data ready | |
| Causal patching / ablation (link brain-aligned tokens to behavior) | ⬜ Planned | |
| Multi-model replication (Qwen2-VL, InternVL2, LLaVA-NeXT) | ⬜ Planned | |
| Teacher-forced captions / caption-embedding probes | ⬜ Planned | Fairer linguisticization test. |

### Predicted reviewer scores (R4, current state)
Novelty 7 · Significance 5 now / 7 potential · Soundness 4–5 now / 7 after P0 · Clarity 7 · Repro 5.
Verdict: strong workshop / promising preprint; main-conference borderline only after P0 + matching-level test.

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

- `main`: synced to working branch as of N=6 results (fast-forward)
- `review/gpt-pro-fixes`: current working branch
- Recent: SemReps N=6 encoding + noise ceiling + significance, R4 reframe (SemReps spine, NSD demoted)
