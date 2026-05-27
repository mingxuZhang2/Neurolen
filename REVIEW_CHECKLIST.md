# GPT-5.5 Pro Review Checklist

Based on detailed review (2026-05-27). Items ordered by priority.
Mark `[x]` when addressed with commit hash.

---

## P0 — Must fix before any submission

### Metrics & Statistical Rigor

- [x] **Replace `best_r` (max-over-voxels) with `mean_r` as primary metric**
  - `best_r` inflates r via multiple-comparison selection bias (2000+ voxels per ROI)
  - Commit: `3382ca4` — F3/F4 now use `mean_r`
  - Result: V1 peak shifted from L29 → L14; "reversed hierarchy" conclusion invalidated

- [ ] **Add noise-ceiling normalization**
  - Raw mean_r (0.09) is hard to interpret; normalize by split-half reliability ceiling
  - NSD provides ncsnr per voxel → compute NC = Spearman-Brown corrected split-half r

- [ ] **Nested CV for voxel selection**
  - If reporting any per-voxel results: select voxels on inner training fold, evaluate on held-out test fold
  - Alternatively: pre-select top-N reliable voxels by ncsnr (independent of model)

- [ ] **Permutation null distribution**
  - Shuffle token-to-image pairing 1000×, compute null mean_r distribution
  - Report p-value per (layer, ROI) and FDR correction

### Baselines

- [ ] **CLIP patch token baseline (most critical)**
  - Extract CLIP-ViT patch output (576×1024) + MLP projector output (576×4096)
  - Run identical spatial encoding → compare with LLaMA L0-L31
  - Code written: `clip_baseline_extract.py`, job 315601 submitted
  - Key question: does LLaMA decoder add brain alignment beyond what CLIP already provides?

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
