# NeuroLens: Technical Design Document

## Overview

NeuroLens maps the layer-wise cross-modal processing stages of MLLMs onto the neuroscience-established brain functional hierarchy, using representational similarity analysis. The project compares three MLLM processing stages (visual encoding, cross-modal fusion, linguistic refinement) against three brain functional systems (ventral visual stream, multimodal integration areas, language network), quantifying alignment with RSA, CKA, and linear predictivity, then validating causally via activation patching.

This document specifies every component needed for implementation: models, datasets, brain ROIs, analysis methods, reusable code, neuroscience hypotheses, causal validation, compute requirements, and risk mitigation.

---

## 1. Model Selection

### 1.1 Primary Models

We use three MLLMs with distinct architectures to test whether stage-brain correspondence is a general phenomenon rather than an artifact of a single design.

#### LLaVA-1.5-7B (Primary)

- **HuggingFace ID**: `liuhaotian/llava-v1.5-7b`
- **Architecture**: CLIP-ViT-L/14-336px vision encoder + Vicuna-7B-v1.5 (LLaMA-2-based) LLM backbone. Two-layer MLP projector maps vision features into the LLM embedding space. Vision tokens are concatenated with text tokens before the LLM.
- **Layer count**: 32 transformer layers in the LLM decoder. CLIP encoder has 24 layers (ViT-L/14).
- **Why chosen**: Most studied MLLM for interpretability. Zhang et al. (CVPR 2025), VisiPruner (EMNLP 2025), FCCT (AAAI 2025), Visual Attention Sink (ICLR 2025), and Yu & Lee (2025) all analyze LLaVA-1.5. The three-stage processing pattern is best documented here. Yu & Lee (2025) explicitly identified: early layers perform visual grounding, middle layers support lexical integration and semantic reasoning, final layers prepare task-specific outputs.
- **Stage boundaries (from prior work)**:
  - Stage 1 (Visual Encoding): Layers 0-10 (approximately). VisiPruner found shallow layers recognize task intent with visual tokens as passive attention sinks.
  - Stage 2 (Cross-Modal Fusion): Layers 11-22 (approximately). VisiPruner found middle layers show abrupt cross-modal fusion driven by few critical visual tokens. FCCT found MHSAs of last token in middle layers aggregate cross-modal information.
  - Stage 3 (Linguistic Refinement): Layers 23-31 (approximately). VisiPruner found deep layers discard vision tokens for linguistic refinement.
- **Activation extraction**: Register forward hooks on each `model.model.layers[i]` to capture hidden states. Shape per layer: `(batch, seq_len, 4096)`. We extract the mean-pooled representation across all visual tokens for visual-stage alignment, across all text tokens for language-stage alignment, and across all tokens for integration-stage alignment.

#### Qwen2-VL-7B-Instruct

- **HuggingFace ID**: `Qwen/Qwen2-VL-7B-Instruct`
- **Architecture**: Native Vision-Language model. Uses a ViT vision encoder with dynamic resolution (Naive Dynamic Resolution). The vision encoder processes images at variable resolutions and produces a variable number of visual tokens. LLM backbone is Qwen2-7B (28 layers, hidden dim 3584).
- **Layer count**: 28 transformer layers in the LLM.
- **Why chosen**: Architecturally distinct from LLaVA (different vision-language coupling mechanism). Yu & Lee (2025) confirmed the same three-stage pattern exists in Qwen2-VL: "while the overall stage-wise structure remains stable across variations in visual tokenization, instruction tuning data, and pretraining corpus." This makes it ideal for testing generality.
- **Stage boundaries**: Need to be empirically determined per model but expected to follow the same early/middle/late pattern per Yu & Lee (2025). Approximate: layers 0-9 / 10-19 / 20-27.
- **Activation extraction**: Same hook-based approach on `model.model.layers[i]`. Hidden dim: 3584.

#### InternVL2-8B

- **HuggingFace ID**: `OpenGVLab/InternVL2-8B`
- **Architecture**: InternViT-300M-448px vision encoder + InternLM2-7B LLM backbone. Uses a pixel shuffle downsampling + MLP projector. Dynamic resolution with max 12 tiles of 448x448.
- **Layer count**: 32 transformer layers in InternLM2-7B.
- **Why chosen**: Third major open MLLM family. Different vision encoder (InternViT vs. CLIP vs. Qwen-ViT), different LLM backbone (InternLM2 vs. LLaMA-2 vs. Qwen2). If the stage-brain correspondence holds across all three, the finding is robust.
- **Stage boundaries**: Approximately layers 0-10 / 11-22 / 23-31 (to be empirically verified).
- **Activation extraction**: Hooks on `model.language_model.model.layers[i]`.

### 1.2 Activation Extraction Strategy

For all models, the extraction procedure is:

1. **Input**: Process image-text pairs from the fMRI stimulus set through the MLLM.
2. **Hook placement**: Register `register_forward_hook` on each decoder layer's output (post-LayerNorm).
3. **Token selection**: For each layer, extract:
   - `h_vis[l]`: Mean-pooled hidden states across visual token positions. Shape: `(n_stimuli, hidden_dim)`.
   - `h_txt[l]`: Mean-pooled hidden states across text token positions. Shape: `(n_stimuli, hidden_dim)`.
   - `h_all[l]`: Mean-pooled hidden states across all token positions. Shape: `(n_stimuli, hidden_dim)`.
4. **Dimensionality reduction**: Before RSA/CKA, apply PCA to reduce to 256 dimensions (following standard practice in brain encoding literature to avoid high-dimensional noise).
5. **Storage**: Save as `numpy` arrays, organized as `activations/{model_name}/layer_{l}_vis.npy`, etc.

### 1.3 Vision Encoder Activations

In addition to LLM decoder layers, we also extract intermediate representations from the vision encoder:

- **LLaVA**: CLIP-ViT-L/14 has 24 layers. Extract CLS token representation at layers [0, 6, 12, 18, 23].
- **Qwen2-VL**: Extract from the Qwen ViT at corresponding depths.
- **InternVL2**: Extract from InternViT-300M at corresponding depths.

These provide the "pure visual" baseline: early vision encoder layers should align best with V1-V3, and later vision encoder layers with V4/IT, independent of any cross-modal processing.

---

## 2. Dataset Selection

### 2.1 Primary Dataset: Natural Scenes Dataset (NSD)

**Reference**: Allen et al. (2021), "A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence", Nature Neuroscience, 610 citations.

**Details**:
- **Download**: https://naturalscenesdataset.org/ (AWS S3 bucket: `s3://natural-scenes-dataset/`)
- **Subjects**: 8 subjects (sub-01 through sub-08), scanned extensively over ~1 year
- **Stimuli**: 73,000 distinct natural scene images from MS-COCO. Each subject viewed 10,000 images (with 1,000 shared across all subjects). Each image was shown 3 times.
- **fMRI resolution**: 7T scanner, 1.8mm isotropic voxels, whole-brain coverage
- **Preprocessing**: NSD provides multiple preprocessing levels:
  - Raw BOLD timeseries
  - GLMsingle beta maps (single-trial response estimates, recommended for our use)
  - **Crucially**: NSD provides pre-computed ROI definitions using the `nsdgeneral` ROI set and FreeSurfer-based anatomical parcellations
- **Brain ROIs available in NSD**:
  - **Early visual cortex**: V1, V2, V3 (retinotopic mapping based)
  - **Ventral stream**: V4 (hV4), ventral occipital cortex (VO), lateral occipital complex (LOC), fusiform face area (FFA), parahippocampal place area (PPA)
  - **Language areas**: NOT directly provided by NSD (NSD is a visual task). Must use FreeSurfer atlas parcellation to define Broca's (pars opercularis + pars triangularis of left IFG) and Wernicke's (posterior STG) areas.
  - **Multimodal integration**: Angular gyrus and STS can be defined from FreeSurfer Destrieux or Glasser atlas parcellations included with NSD.
- **Key advantage**: By far the largest, highest-quality visual fMRI dataset. 610 citations. Images are from MS-COCO, which means MLLMs trained on COCO-like data have likely seen similar distributions.
- **Key limitation**: Visual-only task. Subjects viewed images, not text. Language areas were not specifically activated by the task. However, language areas still respond to visual scenes (semantic processing of scenes activates language regions).

**Data size**: ~1.8TB total. We need only the GLMsingle betas and ROI masks, approximately 50-100GB per subject.

### 2.2 Secondary Dataset: Nikolaus et al. (2024) Paired Image-Text fMRI

**Reference**: Nikolaus et al. (2024), "Modality-Agnostic fMRI Decoding of Vision and Language", arXiv 2403.11771, 5 citations.

**Details**:
- **Download**: The paper states data is available but specific URL needs to be confirmed. The paper comes from CNRS/Universite de Toulouse.
- **Subjects**: Multiple subjects (exact number from paper)
- **Stimuli**: ~8,500 trials per subject. Critically, subjects viewed BOTH images AND text descriptions of the same images. This is exactly what we need for multimodal integration analysis.
- **Key finding from paper**: "High-level visual (temporal) regions perform well on BOTH image and text stimuli", suggesting modality-agnostic representations. This directly supports our hypothesis that MLLM fusion-stage layers should align with these temporal regions.
- **Brain ROIs**: The paper reports results for occipital (low-level visual), temporal (high-level visual / multimodal), and language regions.
- **Key advantage**: PAIRED image-text stimuli. This dataset is uniquely suited for testing our hypothesis because:
  - When subjects view images, we can compare MLLM visual-stage activations to visual cortex
  - When subjects read text, we can compare MLLM language-stage activations to language areas
  - The temporal/parietal regions that respond to BOTH modalities can be compared to MLLM fusion-stage activations
- **Key limitation**: Smaller scale than NSD. Fewer subjects. Less spatial resolution (likely 3T).

### 2.3 Tertiary Dataset: BOLD5000

**Reference**: Chang et al. (2019), "BOLD5000, a public fMRI dataset while viewing 5000 visual images", Scientific Data.

**Details**:
- **Download**: https://bold5000-dataset.github.io/website/
- **Subjects**: 4 subjects
- **Stimuli**: 5,254 images from ImageNet, COCO, and SUN scene databases. 4,916 unique images (113 shared).
- **Brain ROIs**: Whole-brain fMRI. Standard anatomical parcellations available.
- **Use case**: Backup dataset if NSD proves too large to process, or for replication purposes. Wang et al. (arXiv 2024, "Decoding Visual Experience and Mapping Semantics through Whole-Brain Analysis") demonstrated 43% improved semantic accuracy using BOLD5000 with foundation models. Their finding that "the default mode network contributes significantly to stimulus decoding" supports our focus on integration regions.
- **Limitation**: Visual-only task, smaller scale.

### 2.4 Dataset Decision

**Recommended approach**: Use **NSD as primary** and **Nikolaus et al. as validation**.

- NSD provides the scale and quality needed for robust RSA/CKA analysis (10,000 images per subject, 8 subjects, 7T resolution). It is the gold standard.
- Nikolaus et al. provides the paired image-text stimuli needed to test the fusion-stage hypothesis specifically.
- If the Nikolaus dataset proves unavailable or too small, fall back to using NSD alone with the argument that language areas respond to semantic content of visual scenes.

### 2.5 Stimulus Alignment

**Critical detail**: We must ensure the images shown to fMRI subjects are also processable by the MLLMs.

- NSD images come from MS-COCO. COCO images have associated captions. We use the COCO image as visual input and a COCO caption as text input to the MLLM. This creates naturally paired image-text stimuli for the MLLM even when fMRI subjects only viewed images.
- For Nikolaus et al., the paired image-text stimuli are directly usable.

---

## 3. Brain ROI Mapping

### 3.1 Stage-to-Region Mapping

The core hypothesis of NeuroLens is a three-stage correspondence:

| MLLM Stage | MLLM Layers (LLaVA-1.5-7B) | Brain Functional System | Brain ROIs |
|---|---|---|---|
| Stage 1: Visual Encoding | Layers 0-10 | Ventral Visual Stream (early) | V1, V2, V3, hV4 |
| Stage 2: Cross-Modal Fusion | Layers 11-22 | Multimodal Integration Network | STS, Angular Gyrus, TPJ |
| Stage 3: Linguistic Refinement | Layers 23-31 | Language Network | Broca's area (left IFG), Wernicke's area (left pSTG) |

Additionally, for completeness:
- Vision encoder layers should align with V1-V4 in a graded fashion (early encoder layers with V1, late with V4/IT).
- High-level ventral stream areas (FFA, PPA, LOC) should bridge between Stage 1 and Stage 2.

### 3.2 ROI Definitions Using Standard Atlases

We use two atlases, both available through FreeSurfer and included in NSD:

#### Glasser HCP Multi-Modal Parcellation (MMP 1.0)

The Glasser atlas (Glasser et al., Nature 2016, 180 parcels per hemisphere) provides fine-grained, functionally validated parcellations. We use the following parcels:

**Stage 1 (Visual Encoding)**:
- V1 (primary visual cortex): Glasser parcel V1
- V2: Glasser parcel V2
- V3: Glasser parcel V3
- V4 (hV4): Glasser parcel V4
- Aggregate these as `visual_early` ROI.

**Stage 2 (Cross-Modal Fusion)**:
- Superior Temporal Sulcus (STS): Glasser parcels STSdp, STSda, STSvp, STSva. The STS is a key multimodal integration region, responding to both visual and auditory/linguistic stimuli. Lin et al. (Brain and Language 2024) identified left anterior STG and angular gyrus as "connector hubs" where semantic dimensions from multiple modalities overlap.
- Angular Gyrus (AG): Glasser parcels PGi, PGs. The AG is part of the default mode network and is consistently implicated in semantic integration across modalities.
- Temporo-Parietal Junction (TPJ): Glasser parcels TPOJ1, TPOJ2, TPOJ3. The TPJ integrates information from visual, auditory, and somatosensory cortices.
- Aggregate as `multimodal_integration` ROI.

**Stage 3 (Linguistic Refinement)**:
- Broca's area: Glasser parcels 44 (pars opercularis) and 45 (pars triangularis) in left hemisphere. Central to language production and syntactic processing.
- Wernicke's area: Glasser parcels A4 and A5 (auditory areas), plus PSTg, in left hemisphere. Note: the classical "Wernicke's area" is debated; we use posterior STG and surrounding areas.
- Left Inferior Frontal Gyrus (IFG): Glasser parcels IFJa, IFJp, IFSa, IFSp in left hemisphere. Broader language-related frontal region.
- Aggregate as `language_network` ROI.

#### FreeSurfer Destrieux Atlas (Backup)

If the Glasser atlas is not directly available for a subject, we can use the FreeSurfer Destrieux atlas (2009):
- V1: `G_cuneus` + calcarine sulcus regions
- Angular Gyrus: `G_pariet_inf-Angular`
- STS: `S_temporal_sup`
- Broca's: `G_front_inf-Opercular` + `G_front_inf-Triangul` (left hemisphere)
- Wernicke's: `G_temp_sup-Lateral` (posterior portion, left hemisphere)

### 3.3 ROI Data Extraction from NSD

NSD provides several ROI definition methods:
1. **nsdgeneral ROI**: A brain mask covering visually responsive voxels. Too broad for our purposes.
2. **prf-eccentricity and prf-angle**: Population receptive field maps for V1-V3. Use these for precise V1/V2/V3 definitions.
3. **FreeSurfer parcellation**: Available for all subjects. Use `fsaverage` space for group analysis or native space for subject-specific analysis.
4. **Functional localizer ROIs**: NSD includes localizer data for FFA, PPA, EBA, OPA, RSC. Use these for ventral stream specificity.

**Processing pipeline**:
1. Load NSD GLMsingle betas for each subject (already in MNI or native space).
2. Apply ROI masks to extract voxel responses within each ROI.
3. Average across voxels within each ROI to get an ROI-level response vector per stimulus. Alternatively, keep the full voxel pattern for RSA (recommended for higher sensitivity).
4. Construct the brain RDM: for N stimuli, compute the N x N dissimilarity matrix using correlation distance between the voxel patterns.

---

## 4. Method Pipeline

### 4.1 Overview

```
Step 1: Data Preparation
  - Download NSD betas + ROI masks
  - Download COCO images + captions
  - Align: identify which NSD stimuli map to which COCO images

Step 2: MLLM Activation Extraction
  - For each model: process NSD/COCO image-caption pairs
  - Extract per-layer activations (visual tokens, text tokens, all tokens)
  - PCA to 256 dimensions
  - Compute model RDMs per layer

Step 3: Brain RDM Construction
  - For each subject, each ROI: extract voxel patterns per stimulus
  - Compute brain RDMs per ROI (correlation distance)

Step 4: Stage-Region Alignment (RSA)
  - Compare model RDMs (per layer) to brain RDMs (per ROI)
  - Key analysis: which layers align best with which brain regions?
  - Test hypothesis: early layers -> visual cortex, middle -> STS/AG, late -> Broca's/Wernicke's

Step 5: Stage-Region Alignment (CKA)
  - Compute CKA between layer activations and brain voxel patterns
  - Cross-validate with RSA results

Step 6: Encoding Model (Linear Predictivity)
  - Ridge regression: predict brain voxels from MLLM layer activations
  - Cross-validated R^2 per layer per ROI
  - Third metric to triangulate with RSA and CKA

Step 7: Causal Validation (Activation Patching)
  - Patch activations at specific MLLM stages
  - Measure effect on model output
  - Compare failure modes to known brain lesion effects

Step 8: Multi-Model Comparison
  - Repeat Steps 2-7 for all three models
  - Test convergence: do all models show similar stage-brain mapping?
```

### 4.2 Step-by-Step Details

#### Step 1: Data Preparation

```python
# NSD stimulus mapping
# NSD provides a file mapping each trial to a COCO image ID
# File: nsd_stim_info_merged.csv (available in NSD release)
# Columns: nsdId, cocoId, cocoSplit
# We use this to get COCO captions for each NSD stimulus

# Download NSD betas (GLMsingle, version 3)
# Location: s3://natural-scenes-dataset/nsddata_betas/
# Files: betas_fithrf_GLMdenoise_RR/{subject}/betas_session{XX}.nii.gz
# Each session has 750 trials

# Download ROI masks
# Location: s3://natural-scenes-dataset/nsddata/ppdata/{subject}/
# Files: roi/prf-eccentricity.nii.gz, roi/prf-visualrois.nii.gz, etc.
```

For tractability, we start with the **shared 1,000 images** (seen by all 8 subjects). This gives us 1,000 stimuli with 8 independent brain responses each, sufficient for robust RSA.

#### Step 2: Activation Extraction

```python
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor

model = LlavaForConditionalGeneration.from_pretrained(
    "liuhaotian/llava-v1.5-7b",
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("liuhaotian/llava-v1.5-7b")

# Register hooks
activations = {}
def get_hook(name):
    def hook(module, input, output):
        # output is a tuple; first element is hidden states
        activations[name] = output[0].detach().cpu()
    return hook

for i, layer in enumerate(model.language_model.model.layers):
    layer.register_forward_hook(get_hook(f"layer_{i}"))

# Process each stimulus
for img, caption in stimulus_pairs:
    inputs = processor(images=img, text=caption, return_tensors="pt")
    with torch.no_grad():
        model(**inputs)
    # Extract and store activations per layer
    for layer_name, act in activations.items():
        # Separate visual and text token positions
        # visual tokens: positions determined by image token count
        # text tokens: remaining positions
        save_activation(layer_name, act)
```

#### Step 3: Brain RDM Construction

```python
import rsatoolbox

# For each subject and ROI:
# brain_data shape: (n_stimuli, n_voxels)
dataset = rsatoolbox.data.Dataset(brain_data)
rdm_brain = rsatoolbox.rdm.calc_rdm(
    dataset,
    method='correlation'  # 1 - Pearson correlation
)
```

#### Step 4-6: Alignment Analysis

See Section 5 for detailed method specifications.

---

## 5. Analysis Methods

### 5.1 RSA (Representational Similarity Analysis)

**Toolkit**: `rsatoolbox` (https://github.com/rsagroup/rsatoolbox), 242 stars, MIT license, actively maintained (last updated April 2026).

**Procedure**:
1. Compute model RDM for each layer: `rdm_model[l] = rsatoolbox.rdm.calc_rdm(model_activations[l], method='correlation')`
2. Compute brain RDM for each ROI: `rdm_brain[r] = rsatoolbox.rdm.calc_rdm(brain_voxels[r], method='correlation')`
3. Compare: `rsatoolbox.rdm.compare(rdm_model[l], rdm_brain[r], method='corr')` (Spearman correlation between upper triangles)
4. Statistical testing: Use `rsatoolbox.inference.eval_fixed` with bootstrap resampling for confidence intervals.

**Distance metric for RDMs**: Use **correlation distance** (1 - Pearson r) following standard practice in the brain encoding literature (Kriegeskorte et al., 2008). Soni et al. (2024) showed that metric choice impacts conclusions; we report results with correlation distance as primary and Euclidean distance as supplementary to address this concern.

**Key API**:
```python
import rsatoolbox

# Create RDMs
model_rdm = rsatoolbox.rdm.calc_rdm(
    rsatoolbox.data.Dataset(model_acts),  # (n_stimuli, n_features)
    method='correlation'
)
brain_rdm = rsatoolbox.rdm.calc_rdm(
    rsatoolbox.data.Dataset(brain_voxels),  # (n_stimuli, n_voxels)
    method='correlation'
)

# Compare RDMs
similarity = rsatoolbox.rdm.compare(model_rdm, brain_rdm, method='spearman')

# Noise ceiling (upper/lower bounds on achievable RSA)
noise_ceiling = rsatoolbox.inference.eval_fixed(
    model_rdm, brain_rdms_per_subject, method='spearman'
)
```

### 5.2 CKA (Centered Kernel Alignment)

**Implementation**: We implement CKA directly rather than relying on a specific package, as it is a simple formula. We also found `ryusudol/Centered-Kernel-Alignment` (5 stars, efficient implementation).

**Linear CKA formula**:
```
CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
```
where X and Y are centered activation matrices (n_stimuli x n_features).

**Implementation**:
```python
def linear_cka(X, Y):
    """Compute linear CKA between two sets of representations.
    X: (n, p) -- e.g., MLLM layer activations
    Y: (n, q) -- e.g., brain voxel patterns
    """
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    
    hsic_xy = np.linalg.norm(Y.T @ X, 'fro') ** 2
    hsic_xx = np.linalg.norm(X.T @ X, 'fro')
    hsic_yy = np.linalg.norm(Y.T @ Y, 'fro')
    
    return hsic_xy / (hsic_xx * hsic_yy)
```

**Linear vs. Kernel CKA**: We use **linear CKA** as the primary metric. Kornblith et al. (2019) showed linear CKA is more robust to perturbations and easier to interpret. Kernel CKA (with RBF kernel) is reported as supplementary. Following Soni et al. (2024), we use CKA as a complement to RSA, not a replacement.

### 5.3 Linear Predictivity (Encoding Models)

**Method**: Ridge regression from MLLM layer activations to brain voxel responses, cross-validated.

**Procedure**:
1. For layer `l` and ROI `r`:
   - X: MLLM activations at layer l, shape (n_stimuli, 256) after PCA
   - Y: Brain voxel responses in ROI r, shape (n_stimuli, n_voxels)
2. 5-fold cross-validation: Train ridge regression on 4/5 of stimuli, predict on 1/5.
3. Metric: Pearson correlation between predicted and actual voxel responses, averaged across voxels.
4. Ridge regularization: Alpha selected via nested cross-validation from [1e-2, 1e-1, 1, 10, 100, 1000].

**Implementation**:
```python
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
import numpy as np

def linear_predictivity(X, Y, n_folds=5):
    """Predict brain voxels Y from model features X.
    Returns: mean cross-validated Pearson r across voxels.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    correlations = []
    
    for train_idx, test_idx in kf.split(X):
        ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 1000])
        ridge.fit(X[train_idx], Y[train_idx])
        Y_pred = ridge.predict(X[test_idx])
        
        # Per-voxel correlation
        for v in range(Y.shape[1]):
            r = np.corrcoef(Y_pred[:, v], Y[test_idx, v])[0, 1]
            correlations.append(r)
    
    return np.mean(correlations)
```

### 5.4 Addressing the Metric Choice Problem

Soni et al. (2024, "Conclusions about Neural Network to Brain Alignment are Profoundly Impacted by the Similarity Measure", 19 citations) demonstrated that RSA, CKA, and linear predictivity can give divergent conclusions about which model layer best aligns with which brain region.

**Our strategy**:
1. Report all three metrics (RSA, CKA, linear predictivity) for every layer-ROI combination.
2. Declare a "convergent finding" only when all three metrics agree on the stage-region mapping.
3. Where metrics diverge, analyze why (following Wu et al., 2025, "Measuring the Measures: Discriminative Capacity of Representational Similarity Metrics Across Model Families", which found that separability increases as metrics impose more stringent alignment constraints).
4. Use the metric comparison itself as a contribution: which metric is most appropriate for MLLM-brain comparison?

---

## 6. Reusable Artifacts

### 6.1 Software Packages

| Package | Version | Purpose | URL | License |
|---|---|---|---|---|
| `rsatoolbox` | >= 0.1.5 | RSA computation, RDM comparison, noise ceiling | https://github.com/rsagroup/rsatoolbox | MIT |
| `transformers` | >= 4.40 | MLLM loading, tokenization, inference | https://github.com/huggingface/transformers | Apache-2.0 |
| `scikit-learn` | >= 1.3 | Ridge regression, PCA, cross-validation | https://github.com/scikit-learn/scikit-learn | BSD-3 |
| `nibabel` | >= 5.0 | Loading NIfTI brain images | https://github.com/nipy/nibabel | MIT |
| `nilearn` | >= 0.10 | Brain image manipulation, masking, ROI extraction | https://github.com/nilearn/nilearn | New BSD |
| `cortex` (pycortex) | >= 1.2 | Brain visualization, flatmaps | https://github.com/gallantlab/pycortex | BSD-2 |
| `torch` | >= 2.0 | Deep learning backend | https://pytorch.org | BSD-3 |
| `pycocotools` | >= 2.0 | COCO dataset API for stimulus alignment | https://github.com/cocodataset/cocoapi | BSD-2 |

### 6.2 Existing Codebases to Leverage

#### Algonauts Project 2023 Challenge Code

- **URL**: https://colab.research.google.com/github/Algonauts2023/algonauts2023_tutorial/blob/main/algonauts_2023_tutorial_challenge.ipynb (tutorial notebook)
- **What we take**: NSD data loading pipeline, ROI extraction code, encoding model evaluation framework. The Algonauts 2023 challenge (Gifford et al., 2023, 54 citations) was built on NSD and provides well-tested infrastructure for brain encoding evaluation.

#### nsd_access / nsd_mapdata

- **URL**: https://github.com/tknapen/nsd_access (if available) or direct NSD code from the NSD release
- **What we take**: Helper functions for loading NSD betas, mapping stimuli to COCO IDs, extracting ROI data.

#### Oota et al. (ICLR 2025) Brain Encoding Pipeline

- **Reference**: "Correlating instruction-tuning (in multimodal models) with vision-language processing (in the brain)", 6 citations.
- **Code**: https://github.com/subbareddy248/mllm_videos (mentioned in their ICLR 2025 follow-up paper)
- **What we take**: Their MLLM-to-brain encoding pipeline, which already extracts MLLM embeddings and predicts fMRI responses. This is the closest existing code to what we need. Modifications required: add layer-wise extraction, add RSA/CKA comparison, add ROI-specific analysis.

#### Brain-Score Toolkit

- **URL**: https://github.com/brain-score/brain-score (well-known benchmark)
- **What we take**: Reference for how brain alignment is properly evaluated. We do NOT use Brain-Score directly (it is designed for vision models, not MLLMs), but follow their methodological standards: noise ceiling estimation, cross-validated encoding, ceiling-normalized scores.

#### Yu & Lee (2025) Probing Framework

- **Reference**: "How Multimodal LLMs Solve Image Tasks" (10 citations, arXiv 2508.20279)
- **What we take**: Their layer-wise probing methodology for LLaVA, LLaVA-Next, and Qwen2-VL. Their code provides activation extraction and stage identification for these specific models.

### 6.3 NSD Data Utilities

The NSD provides several Python utilities:
- `nsd_mapdata.py`: Maps data between different NSD coordinate spaces
- `nsd_datalad`: DataLad-based download manager
- The Algonauts 2023 challenge provides Google Colab notebooks with complete NSD loading pipelines

---

## 7. Neuroscience Hypotheses

### 7.1 Primary Hypotheses (Convergent Mapping)

For each hypothesis, we state: the MLLM prediction, the neuroscience basis, and the specific testable comparison.

#### H1: Visual Encoding Stage Aligns with Ventral Visual Stream

- **Prediction**: MLLM layers 0-10 (Stage 1) show highest RSA/CKA with V1, V2, V3, V4.
- **Neuroscience basis**: The ventral visual stream (V1 -> V2 -> V4 -> IT) processes visual features in a hierarchy from simple edges (V1) to complex object representations (IT). DNN early layers are known to align with V1-V3 (Yamins et al., 2014; Khaligh-Razavi & Kriegeskorte, 2014). The ventral stream hierarchy has been the most replicated finding in computational neuroscience.
- **Specific test**: RSA(layer_l, V1) > RSA(layer_l, STS) for layers l in [0,10]. The peak alignment layer should be earlier for V1 than for V4.
- **Expected pattern**: Graded alignment: V1 peaks at very early layers, V2 at slightly later, V3/V4 at the end of Stage 1. This gradient within Stage 1 mirrors the hierarchical processing depth along the ventral stream.

#### H2: Cross-Modal Fusion Stage Aligns with Multimodal Integration Regions

- **Prediction**: MLLM layers 11-22 (Stage 2) show highest RSA/CKA with STS, angular gyrus, and TPJ.
- **Neuroscience basis**: STS and angular gyrus are established multimodal integration hubs. Lin et al. (Brain and Language 2024) showed that "neural correlates of semantic dimensions overlap in default mode network (left anterior STG and angular gyrus), acting as connector hubs." Nikolaus et al. (2024) found that "high-level visual (temporal) regions perform well on BOTH image and text stimuli", demonstrating modality-agnostic processing. Subramaniam et al. (ICML 2024, 19 citations) used multimodal DNNs to identify sites of multimodal integration in the brain via SEEG, finding CLIP-style training best predicts neural activity at integration sites.
- **Specific test**: RSA(layer_l, STS) > RSA(layer_l, V1) AND RSA(layer_l, STS) > RSA(layer_l, Broca) for layers l in [11,22].
- **Expected pattern**: Integration regions should peak in alignment at the layers where VisiPruner observed "abrupt cross-modal fusion."

#### H3: Linguistic Refinement Stage Aligns with Language Network

- **Prediction**: MLLM layers 23-31 (Stage 3) show highest RSA/CKA with Broca's and Wernicke's areas.
- **Neuroscience basis**: Broca's area (left IFG) is involved in language production and syntactic processing. Wernicke's area (posterior STG) is involved in language comprehension. Goldstein et al. (Nature Communications 2025, 7 citations) showed that "deeper LLM layers correspond to later brain activity, particularly in Broca's area and other language-related regions." Mischler et al. (Nature Machine Intelligence 2024, 77 citations) found that high-performing LLMs "align more closely with the hierarchical feature extraction pathways of the brain." Lei et al. (AAAI 2025, 2 citations) showed that "improvements in LLM performance drive representational architectures toward brain-like hierarchies, especially at higher semantic abstraction levels."
- **Specific test**: RSA(layer_l, Broca) > RSA(layer_l, V1) for layers l in [23,31].
- **Expected pattern**: Language area alignment should increase monotonically through Stage 3.

#### H4: Cross-Model Convergence

- **Prediction**: The stage-region mapping (H1-H3) holds across LLaVA-1.5, Qwen2-VL, and InternVL2 despite architectural differences.
- **Neuroscience basis**: Mischler et al. (2024) found convergent feature extraction hierarchies across LLMs and the brain. Yu & Lee (2025) found the three-stage structure is stable across MLLM architectures. If this convergence extends to brain alignment, it suggests the stage structure is a computational necessity rather than an architectural artifact.
- **Specific test**: Rank correlation of layer-brain-alignment profiles across models should be significantly positive (Spearman rho > 0.5).

### 7.2 Secondary Hypotheses (Fine-Grained Predictions)

#### H5: Within-Stage Gradient Mirrors Within-Region Hierarchy

- **Prediction**: Within Stage 1, the peak alignment layer for V1 < V2 < V3 < V4, mirroring the hierarchical depth of the ventral stream.
- **Test**: Compare peak-alignment layer indices across V1, V2, V3, V4.

#### H6: Vision Encoder Layers Provide Stronger V1-V3 Alignment Than LLM Layers

- **Prediction**: CLIP/ViT encoder early layers align better with V1-V3 than any LLM decoder layer, since vision encoder layers are dedicated to visual feature extraction.
- **Test**: max(RSA(encoder_l, V1)) > max(RSA(decoder_l, V1)) for all l.

#### H7: Multimodal Processing Creates Unique Integration Representations

- **Prediction**: Stage 2 alignment with STS/AG is higher when the MLLM processes image+text together than when processing image-only or text-only.
- **Test** (requires Nikolaus dataset or controlled input): Compare RSA(fusion_layers, STS) under three conditions: image+text input, image-only input, text-only input.

### 7.3 Null Hypothesis Control

To ensure our findings are not trivially explained by dimensionality or other confounds:

- **Control 1**: Shuffled-layer control. Randomly shuffle layer assignments and show that the stage-brain mapping breaks.
- **Control 2**: Shuffled-ROI control. Randomly reassign brain ROIs and show no systematic mapping emerges.
- **Control 3**: Untrained model control. Extract activations from an untrained (randomly initialized) LLaVA model. If the mapping disappears, it is driven by learning, not architecture.
- **Control 4**: Noise ceiling comparison. Report all alignments as fraction of noise ceiling (maximum achievable alignment given fMRI noise).

---

## 8. Causal Validation Plan

### 8.1 Motivation

Correlational alignment (RSA/CKA) shows that representations are similar but does not prove functional equivalence. Causal validation via activation patching tests whether disrupting specific MLLM stages causes failures that parallel known effects of brain region damage.

### 8.2 Patching Strategy

We cannot directly use TransformerLens for LLaVA because TransformerLens is designed for autoregressive LLMs and does not natively handle the visual encoder + projector + LLM architecture. Instead, we implement custom activation patching following the methodology of Zhang & Nanda (2023, "Towards Best Practices of Activation Patching", 236 citations).

**Implementation approach**:

```python
def patch_activations(model, inputs, layer_range, patch_type='zero'):
    """
    Patch activations at specified layers.
    layer_range: tuple (start, end) of layer indices to patch
    patch_type: 'zero' (zero ablation), 'mean' (mean ablation), 'noise' (Gaussian noise)
    """
    hooks = []
    
    def patch_hook(module, input, output):
        hidden = output[0]
        if patch_type == 'zero':
            return (torch.zeros_like(hidden),) + output[1:]
        elif patch_type == 'mean':
            return (hidden.mean(dim=0, keepdim=True).expand_as(hidden),) + output[1:]
        elif patch_type == 'noise':
            noise = torch.randn_like(hidden) * hidden.std()
            return (hidden + noise,) + output[1:]
    
    for i in range(layer_range[0], layer_range[1] + 1):
        h = model.language_model.model.layers[i].register_forward_hook(patch_hook)
        hooks.append(h)
    
    with torch.no_grad():
        output = model(**inputs)
    
    for h in hooks:
        h.remove()
    
    return output
```

**Patching method**: Use **mean ablation** as primary (replacing activations with the mean across the dataset), following Zhang & Nanda (2023) who recommend it over zero ablation for being less extreme. Report zero ablation as supplementary.

### 8.3 Causal Predictions

#### Patching Stage 1 (Visual Encoding, Layers 0-10)

- **Expected effect**: Model loses ability to identify visual features. Outputs become generic or unrelated to image content.
- **Brain analog**: Damage to V1-V4 causes cortical blindness or visual agnosia (inability to recognize objects despite intact eyes).
- **Measurement**: On VQA tasks requiring visual recognition (e.g., "What object is in the center?"), measure accuracy drop.

#### Patching Stage 2 (Cross-Modal Fusion, Layers 11-22)

- **Expected effect**: Model can describe visual features AND use language, but fails to bind them correctly. Produces attribute-swap errors (e.g., "the red cat" when the cat is blue but a nearby object is red).
- **Brain analog**: Damage to parietal/temporal integration areas (especially TPJ) causes Balint's syndrome features, including the binding problem: patients see individual features correctly but cannot bind them to the right objects. Campbell et al. (NeurIPS 2024, 69 citations) showed VLM failures already mirror this pattern.
- **Measurement**: On binding-demanding tasks (e.g., "What color is the cat?"), measure attribute-swap error rate specifically.

#### Patching Stage 3 (Linguistic Refinement, Layers 23-31)

- **Expected effect**: Model correctly processes visual information and binds features to objects, but produces linguistically degraded outputs (short, grammatically poor, or incoherent).
- **Brain analog**: Damage to Broca's area causes non-fluent (expressive) aphasia: patients understand visual scenes but cannot produce coherent language descriptions.
- **Measurement**: On open-ended captioning, measure linguistic quality metrics (perplexity, fluency, grammaticality) while tracking whether visual content is correctly identified.

### 8.4 Double Dissociation Test

The strongest causal evidence would be a **double dissociation**:
1. Patching Stage 1 impairs visual recognition more than binding accuracy.
2. Patching Stage 2 impairs binding accuracy more than visual recognition.
3. Patching Stage 3 impairs language fluency more than visual recognition or binding.

If all three dissociations hold, the causal evidence for stage-specific function is strong.

---

## 9. Compute Requirements

### 9.1 Activation Extraction

| Task | Model | Stimuli | GPU | Time | Storage |
|---|---|---|---|---|---|
| Extract all-layer activations | LLaVA-1.5-7B | 1,000 images | 1x H100 (80GB) | ~2 hours | ~20 GB |
| Extract all-layer activations | Qwen2-VL-7B | 1,000 images | 1x H100 (80GB) | ~2 hours | ~18 GB |
| Extract all-layer activations | InternVL2-8B | 1,000 images | 1x H100 (80GB) | ~2 hours | ~20 GB |
| Extract vision encoder activations | All 3 models | 1,000 images | 1x H100 | ~1 hour | ~5 GB |

**Subtotal**: ~7 GPU-hours, ~63 GB storage.

### 9.2 Brain Data Processing

| Task | Data | CPU/GPU | Time | Storage |
|---|---|---|---|---|
| Download NSD betas (8 subjects, shared 1000) | S3 download | CPU | ~4 hours | ~100 GB |
| ROI mask extraction | FreeSurfer + nilearn | CPU | ~2 hours | ~5 GB |
| Compute brain RDMs (1000x1000 per ROI per subject) | CPU | CPU (parallel) | ~1 hour | ~2 GB |

**Subtotal**: CPU-only, ~107 GB storage.

### 9.3 Alignment Analysis

| Task | Data | CPU/GPU | Time |
|---|---|---|---|
| RSA: 32 layers x 10 ROIs x 8 subjects x 3 models | CPU | CPU (parallel) | ~4 hours |
| CKA: 32 layers x 10 ROIs x 8 subjects x 3 models | CPU | CPU (parallel) | ~4 hours |
| Linear predictivity: 32 layers x 10 ROIs x 8 subjects x 3 models, 5-fold CV | CPU | CPU (parallel) | ~12 hours |
| Statistical testing and bootstrap CI | CPU | CPU (parallel) | ~4 hours |

**Subtotal**: ~24 CPU-hours.

### 9.4 Activation Patching

| Task | Model | Conditions | GPU | Time |
|---|---|---|---|---|
| Patch Stage 1 + evaluate | LLaVA-1.5-7B | 3 patch types x 500 eval samples | 1x H100 | ~3 hours |
| Patch Stage 2 + evaluate | LLaVA-1.5-7B | 3 patch types x 500 eval samples | 1x H100 | ~3 hours |
| Patch Stage 3 + evaluate | LLaVA-1.5-7B | 3 patch types x 500 eval samples | 1x H100 | ~3 hours |
| Repeat for Qwen2-VL + InternVL2 | 2 models | Same as above | 2x H100 | ~18 hours |

**Subtotal**: ~27 GPU-hours.

### 9.5 Total

- **GPU**: ~34 H100-hours (~1.5 days on a single H100, or <1 day with 2 GPUs)
- **CPU**: ~24 hours of parallel CPU work
- **Storage**: ~170 GB
- **Download bandwidth**: ~100 GB NSD data + ~30 GB model weights (already cached on HPC3 if common models)

This is very manageable for HPC3 resources.

---

## 10. Risk Mitigation

### Risk 1: Language areas show weak fMRI response in NSD (visual-only task)

- **Likelihood**: Medium. NSD subjects viewed images, not text, so language areas were not primarily engaged.
- **Impact**: H3 (linguistic refinement -> language network) may not hold.
- **Mitigation 1**: Use the Nikolaus et al. dataset, which has paired image-text stimuli and explicitly reports language area activations.
- **Mitigation 2**: Even in visual tasks, language areas are activated by semantic processing of scenes (e.g., interpreting object names, scene categories). Multiple studies show NSD data contains decodable semantic information from temporal and frontal areas.
- **Mitigation 3**: If language areas show low RSA overall, report this as a meaningful finding: the stage-brain mapping holds for visual and integration stages but breaks down for the linguistic stage, suggesting MLLM linguistic processing may diverge from brain language processing.

### Risk 2: Stage boundaries are not sharp, making layer-to-region assignment ambiguous

- **Likelihood**: Medium. Prior work shows gradual transitions, not sharp boundaries.
- **Impact**: Results may show a gradual gradient rather than discrete stages.
- **Mitigation**: Do NOT assume fixed stage boundaries. Instead, report alignment as a continuous function of layer index. Plot alignment heatmaps (layers x ROIs) and let the data reveal whether boundaries are sharp or gradual. A gradient is actually a more interesting and nuanced finding than discrete stages.

### Risk 3: RSA/CKA results diverge (the Soni et al. problem)

- **Likelihood**: High (Soni et al. explicitly showed this happens).
- **Impact**: Cannot make a clear claim about stage-brain mapping.
- **Mitigation**: Use three metrics (RSA, CKA, linear predictivity) and report convergent findings only. The divergence itself becomes a methodological contribution. Follow Wu et al. (2025) analysis of metric discriminative capacity.

### Risk 4: NSD download fails or is too large for HPC3 storage

- **Likelihood**: Low (NSD is well-maintained on AWS S3).
- **Impact**: Cannot proceed with primary dataset.
- **Mitigation**: Start with BOLD5000 (much smaller, ~10 GB). If NSD works, use it; if not, BOLD5000 provides sufficient data for a preliminary analysis.

### Risk 5: Nikolaus et al. dataset is not publicly available

- **Likelihood**: Medium (paper says available but URL unclear).
- **Impact**: Cannot test H7 (multimodal processing uniqueness) or H3 with text stimuli.
- **Mitigation**: Contact the authors (CNRS/Universite de Toulouse). If unavailable, proceed with NSD only and frame the text-stimulus analysis as future work.

### Risk 6: Activation patching on MLLMs does not produce clean dissociations

- **Likelihood**: Medium. MLLMs are complex; patching may produce diffuse effects.
- **Impact**: Causal validation (Section 8) is weakened.
- **Mitigation 1**: Use multiple patch types (zero, mean, noise) and report the cleanest results.
- **Mitigation 2**: Following the Dual-Pathway Circuits paper (Liu et al., 2026) and Nooralahzadeh et al. (2026), use full-sequence activation patching rather than single-token patching, as Nooralahzadeh showed "standard last-token interventions in LLM interpretability do not affect VLMs. In contrast, replacing the full token sequence at layers identified by MAC alters 60-84% of outputs."
- **Mitigation 3**: The causal validation is a bonus component. The paper's primary contribution (stage-brain alignment via RSA/CKA) stands without it. If patching results are unclear, present them as preliminary and leave deeper causal analysis for follow-up work.

### Risk 7: Results show no significant stage-brain correspondence

- **Likelihood**: Low (given the strong prior evidence from Goldstein et al. 2025, Mischler et al. 2024, and established DNN-brain alignment literature).
- **Impact**: No positive finding.
- **Mitigation**: This is actually publishable as a negative result ("MLLMs process cross-modal information differently from the brain despite surface-level stage similarity"). Given the growing interest in NeuroAI, understanding where the brain-model analogy breaks down is valuable.

---

## 11. Implementation Timeline

| Week | Task | Deliverable |
|---|---|---|
| 1 | Download NSD data, set up data pipeline, implement activation extraction | Working extraction code for all 3 models |
| 2 | Extract activations, compute model RDMs, process brain data, extract ROIs | All activation files + brain RDMs |
| 3 | Implement and run RSA, CKA, linear predictivity | Alignment heatmaps, preliminary results |
| 4 | Implement activation patching, run causal experiments | Patching results per stage |
| 5 | Multi-model comparison, statistical testing, figure generation | All results tables and figures |
| 6 | Analyze results, write paper draft | Paper draft |

---

## 12. File Structure

```
mllm/
  implementation/
    DESIGN.md                  # This document
    environment.yml            # Conda environment spec
    src/
      data/
        nsd_loader.py          # NSD data loading and ROI extraction
        stimulus_align.py      # Align NSD stimuli with COCO captions
        bold5000_loader.py     # BOLD5000 loading (backup)
      models/
        extract_activations.py # Hook-based activation extraction for all MLLMs
        model_rdm.py           # Compute model RDMs from activations
      analysis/
        rsa_analysis.py        # RSA computation using rsatoolbox
        cka_analysis.py        # CKA computation
        encoding_models.py     # Linear predictivity (ridge regression)
        statistical_tests.py   # Bootstrap CI, noise ceiling, significance
      patching/
        activation_patching.py # Stage-specific activation patching
        eval_patching.py       # Evaluate patching effects (VQA, captioning)
      visualization/
        heatmaps.py            # Layer x ROI alignment heatmaps
        brain_maps.py          # Brain surface visualizations (pycortex)
    configs/
      models.yaml              # Model HuggingFace IDs, layer counts, stage boundaries
      rois.yaml                # ROI definitions per atlas
      experiments.yaml         # Experiment configurations
    scripts/
      run_extraction.sh        # HPC3 job script for activation extraction
      run_analysis.sh          # HPC3 job script for alignment analysis
      run_patching.sh          # HPC3 job script for activation patching
  survey/                      # Existing survey outputs
  paper/                       # Paper drafts (Stage 3)
```

---

## 13. Key References

1. Allen et al. (2021). A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence. Nature Neuroscience. [NSD dataset]
2. Gifford et al. (2023). The Algonauts Project 2023 Challenge. arXiv 2301.03198. [NSD benchmark infrastructure]
3. Nikolaus et al. (2024). Modality-Agnostic fMRI Decoding of Vision and Language. arXiv 2403.11771. [Paired image-text fMRI]
4. Soni et al. (2024). Conclusions about Neural Network to Brain Alignment are Profoundly Impacted by the Similarity Measure. bioRxiv. [Metric choice matters]
5. Wu et al. (2025). Measuring the Measures: Discriminative Capacity of Representational Similarity Metrics. arXiv 2509.04622. [Metric comparison]
6. Goldstein et al. (2025). Temporal structure of natural language processing in the human brain. Nature Communications. [Layer-to-temporal hierarchy]
7. Mischler et al. (2024). Contextual feature extraction hierarchies converge in LLMs and the brain. Nature Machine Intelligence, 77 citations. [Brain-LLM convergence]
8. Lei et al. (2025). Do Large Language Models Think like the Brain? AAAI. [LLM-brain hierarchical alignment]
9. Oota et al. (2025). Correlating instruction-tuning with vision-language processing. ICLR. [MLLM brain alignment]
10. Tang et al. (2023). Brain encoding models based on multimodal transformers. NeurIPS. [Cross-modal brain encoding]
11. Yu & Lee (2025). How Multimodal LLMs Solve Image Tasks. arXiv 2508.20279. [Three-stage MLLM processing]
12. VisiPruner (2025). EMNLP. [Three-stage cross-modal interaction]
13. Zhang et al. (2025). Cross-modal Information Flow in MLLMs. CVPR. [Two-stage visual-linguistic integration]
14. FCCT (2025). Fine-grained Cross-modal Causal Tracing. AAAI. [Middle-layer cross-modal aggregation]
15. Campbell et al. (2024). The Binding Problem in VLMs. NeurIPS, 69 citations. [Binding problem in VLMs]
16. Zhang & Nanda (2023). Towards Best Practices of Activation Patching. ICLR, 236 citations. [Patching methodology]
17. Nooralahzadeh et al. (2026). Arbitration Failure in VLMs. [Full-sequence patching for VLMs]
18. Liu et al. (2026). Dual-Pathway Circuits of Object Hallucination in VLMs. [Circuit analysis methodology]
19. Lin et al. (2024). Neural correlates of semantic dimensions in default mode network. Brain and Language. [STS/AG as connector hubs]
20. Subramaniam et al. (2024). Multimodal DNN integration sites in brain via SEEG. ICML, 19 citations. [CLIP predicts integration sites]
