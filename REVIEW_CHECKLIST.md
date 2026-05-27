# GPT-5.5 Pro Review Checklist

Based on detailed review (2026-05-27). Items ordered by priority.
Mark `[x]` when addressed with commit hash.

---

## Experimental Results Summary (2026-05-27)

### Completed experiments
1. **Token-level spatial encoding** (576 tokens × 32 layers × V1/V2/V3/V4) ✅
2. **Token-level semantic encoding** via logit lens (576 × 32 × Broca/IFG/auditory/temporal) ✅
3. **CLIP baseline** (CLIP-ViT patches + MLP projector) ✅
4. **Noise ceiling** computation ✅
5. **Parametric significance** test ✅
6. **Permutation null** test 🔄 running

### Key quantitative results

**Spatial encoding (mean_r, averaged over 576 tokens)**:
| Stage | V1 | V2 | V3 | V4 |
|---|---|---|---|---|
| CLIP-ViT patches | 0.048 | 0.047 | 0.079 | 0.072 |
| MLP projector | 0.042 | 0.040 | 0.063 | 0.060 |
| LLaMA L0 | 0.043 | 0.040 | 0.064 | 0.060 |
| **LLaMA L14 (peak)** | **0.090** | **0.094** | **0.135** | **0.132** |
| LLaMA L31 | 0.075 | 0.086 | 0.111 | 0.108 |

**Decoder gain over CLIP**: +72% (V3) to +98% (V2)

**Semantic encoding (mean_r, logit-lens → language ROIs)**:
All values ≈ 0.001–0.008 — **effectively zero**. Vision tokens do NOT become linguistic.

**Noise ceiling (NC_r from NSD ncsnr)**:
| ROI | NC_r | LLaMA peak / NC |
|---|---|---|
| V1 | 0.544 | 16.6% |
| V3 | 0.688 | 19.7% |
| Broca | 0.303 | — (semantic ≈ 0) |
| temporal_pole | 0.111 | — |

**Significance**: all spatial (layer, ROI) t > 44, p < 10⁻¹⁹⁰

### Revised narrative
Original hypothesis: "vision tokens transition from visual to linguistic code across layers"
**Result**: vision tokens remain visual throughout all 32 layers. Decoder mid-layers reshape them into MORE brain-aligned visual representations (+72-98% vs CLIP), but they never acquire linguistic properties measurable by logit-lens → language ROI encoding. Vision-to-language transformation likely occurs at text token positions via cross-attention.

---

## P0 — Must fix before any submission

### Metrics & Statistical Rigor

- [x] **Replace `best_r` (max-over-voxels) with `mean_r` as primary metric**
  - `best_r` inflates r via multiple-comparison selection bias (2000+ voxels per ROI)
  - Commit: `3382ca4` — F3/F4 now use `mean_r`
  - Result: V1 peak shifted from L29 → L14; "reversed hierarchy" conclusion invalidated

- [x] **Add noise-ceiling normalization** — Job 315703 COMPLETED
  - NC_r computed from NSD ncsnr: V1=0.54, V2=0.55, V3=0.69, V4=0.64
  - LLaMA L14 (peak) explains 17-21% of noise ceiling; CLIP patches only 9-11%
  - Language ROIs have low NC (Broca=0.30, temporal_pole=0.11) → confirms low SNR for viewing task
  - Saved: `results_v2/statistical_tests/noise_ceiling.npz`

- [ ] **Nested CV for voxel selection**
  - If reporting any per-voxel results: select voxels on inner training fold, evaluate on held-out test fold
  - Alternatively: pre-select top-N reliable voxels by ncsnr (independent of model)

- [x] **Parametric significance**: t-test across 576 tokens, all p < 1e-190, t > 44
- [ ] **Permutation null distribution** — Job 315704 RUNNING (~2h)
  - 200 image-shuffle permutations × 10 layers × 4 ROIs
  - Waiting for results

### Baselines

- [x] **CLIP patch token baseline (most critical)** — Job 315601 COMPLETED
  - CLIP-ViT patches: V1=0.048, V3=0.079
  - MLP projector: V1=0.042, V3=0.063 (slightly worse than CLIP!)
  - LLaMA L14 peak: V1=0.090, V3=0.135 (+72-98% gain over CLIP)
  - **Conclusion: decoder nearly doubles brain alignment; not just CLIP passthrough**

- [ ] **Low-level visual feature baseline**
  - Gabor bank / edge energy / spatial frequency / color histograms per patch
  - Ridge → V1/V2 to prove encoding captures more than simple pixel statistics

- [ ] **CLIP whole-image embedding baseline**
  - Single CLS token encoding → ROIs (no token-level structure)
  - Proves token-level analysis adds value over image-level

---

## P1 — Required for Nature-level submission

### Interpretation Corrections

- [ ] **Fix "each token is a local image patch" framing**
  - CLIP-ViT self-attention means each token already has global context
  - Correct framing: "spatially-indexed, globally-contextualized visual tokens"
  - Add control: shuffle token positions → does spatial structure disappear?

- [ ] **Weaken "MLLM reverses brain hierarchy" claim**
  - `mean_r` shows all ROIs peak at L13-14 (no reversal)
  - New framing: "LLaMA decoder shows mid-layer peak for all visual ROIs, with V3/V4 slightly earlier than V1/V2"
  - Need low-level feature decomposition to confirm what drives late-layer V1/V2 signal

- [ ] **Fix language ROI interpretation**
  - NSD is a viewing task, not language production → Broca activation is not language-specific
  - Rename "language ROIs" → "visual-semantic / language-adjacent network"
  - Expand ROI set: add FFA, PPA, EBA, STS, AG, ATL, RSC

### Logit Lens Improvements

- [ ] **Add caption-embedding probe as alternative semantic readout**
  - Train linear probe: image-token hidden state → COCO caption MPNet embedding
  - More reliable than raw logit-lens vocabulary projection

- [ ] **Logit-lens sanity checks**
  - Filter out stop words, punctuation, subword fragments from top-K
  - Report noun/verb ratio evolution across layers
  - Random LM-head control: prove not any projection works

- [ ] **Consider Tuned Lens**
  - Train per-layer affine translator to final vocab space (Belrose et al. 2023)
  - More stable than raw logit lens at early layers

### Data Completeness

- [ ] **8 subjects (currently only subj01)**
  - subj02-08 betas downloading (Task #12)
  - Statistical unit must be subject, not voxel/token

- [ ] **Both hemispheres**
  - Currently LH only → cannot verify contralateral retinotopy
  - Need RH betas for subj01+

- [ ] **Full shared1000 (currently 602/1000 due to missing sessions)**
  - Need all 40 sessions per subject for complete coverage

---

## P2 — Strengthens paper significantly

### Cross-Model Generality

- [ ] **Qwen2-VL-7B**
  - Different vision encoder (SigLIP vs CLIP) → tests architecture dependence
  
- [ ] **InternVL2-8B**
  - Different projector design → tests projector contribution

- [ ] **Compare MLLM architectures systematically**
  - Same token-level pipeline across 3+ models
  - Which architectural choices produce more brain-like trajectories?

### Causal Evidence

- [ ] **Token-level ablation linked to brain alignment**
  - Ablate high-V4/FFA-aligned tokens → object recognition drops?
  - Ablate high-V1/V2-aligned tokens → spatial detail/color description drops?
  - Links brain alignment to functional importance

- [ ] **Activation patching between clean/corrupted images**
  - Replace layer L tokens from corrupted image with clean → does behavior + brain-alignment recover?

### Spatial Topology (Retinotopy)

- [ ] **pRF-based validation**
  - NSD provides pRF data (eccentricity, polar angle per voxel)
  - Test: token grid position → best-voxel pRF coordinates correspondence
  - Metrics: eccentricity correlation, polar-angle correlation, contralateral bias

- [ ] **Topographic fidelity metric**
  - token position vs cortical pRF center distance
  - Adjacent tokens → adjacent cortical locations?

### Variance Partitioning (Cross-Modal)

- [ ] **Fix cross-validated R² (not just Pearson r) for variance partition**
- [ ] **PCA must be fit only on training fold**
- [ ] **Nested ridge alpha tuning for concatenated model**
- [ ] **Residualization approach**: text_unique = text residualized vs vision

---

## P3 — Nice to have

- [ ] **Ablation decomposition**: attention-only vs MLP-only vs full layer
- [ ] **Compare image-token vs text-token vs generated-token brain alignment**
  - "Language-like" signal may emerge in generated tokens, not image tokens
- [ ] **Multiple prompt conditions**
  - "Describe this image" vs "What objects are here?" vs "What colors do you see?"
- [ ] **Behavioral metrics upgrade for ablation**
  - Replace GPT-2 perplexity with CLIPScore, SPICE, CIDEr, object recall

---

## Status Key

Completed items reference their commit hash for GPT Pro re-review.
`[x]` = done, `[ ]` = pending.

Last updated: 2026-05-27
