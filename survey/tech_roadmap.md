# Technology Roadmap: Interpretability of Vision-Text Modality Interaction in MLLMs from a Neuroscience Perspective

## 1. MLLM Internal Mechanism Analysis (Track A)

### 1.1 Cross-Modal Information Flow
- **Zhang et al. (CVPR 2025)** "Cross-modal Information Flow in MLLMs" [45 citations]: First systematic study showing TWO distinct stages of visual-linguistic integration in LLaVA models: (1) lower layers transfer general visual features to question tokens; (2) middle layers transfer object-specific visual information to relevant question token positions; (3) higher layers propagate multimodal representation to final position.
- **VisiPruner (EMNLP 2025)** [11 citations]: Identified a THREE-stage cross-modal interaction: shallow layers recognize task intent with visual tokens as passive attention sinks; middle layers show abrupt cross-modal fusion driven by few critical visual tokens; deep layers discard vision tokens for linguistic refinement.
- **Sail (ICCV 2025)** [25 citations]: Demonstrated that removing pretrained ViT components results in significantly different cross-modal information flow patterns.
- **FCCT (AAAI 2025)** [13 citations]: Fine-grained Cross-modal Causal Tracing showing MHSAs of last token in middle layers aggregate cross-modal information; FFNs exhibit three-stage hierarchical progression for visual object representations.

### 1.2 Visual Attention Patterns and Sinks
- **Visual Attention Sink (ICLR 2025)** [102 citations]: Discovered that LMMs consistently allocate high attention to specific irrelevant visual tokens (visual attention sinks), analogous to attention sinks in LLMs. These arise from massive activation of hidden state dimensions.
- **DAMRO (EMNLP 2024)** [49 citations]: Found attention distribution of LLM decoder on image tokens is highly consistent with visual encoder, both focusing on background tokens rather than referred objects.
- **LLaVA-Prumerge (ICCV 2024)** [325 citations]: Exploited sparsity in visual encoder attention scores between CLS token and visual tokens for token reduction.
- **Visual Token Withdrawal (AAAI 2025)** [127 citations]: Discovered information migration: visual information transfers to text tokens within first few layers, making vision tokens unnecessary in deep layers.

### 1.3 Causal Tracing and Circuit Discovery
- **BLIP Causal Tracing (ICCV-W 2023)** [57 citations]: First adaptation of causal tracing to vision-language models (BLIP), showing later layer representations are causally relevant.
- **Circuit Tracing in VLMs (arXiv 2026)** [1 citation]: First framework for transparent circuit tracing in VLMs using transcoders and attribution graphs, uncovering hierarchical visual-semantic integration circuits.
- **Dual-Pathway Circuits (2026)** [0 citations]: Identified visual grounding pathway (correct) and hallucination pathway (erroneous) via activation patching across 5 VLMs.
- **Sparse Visual Thought Circuits (2026)** [0 citations]: Applied SAEs to Qwen3-VL, found mid-decoder locus for task type information, but revealed non-modular circuit interference.
- **Arbitration Failure (2026)** [1 citation]: Showed VLMs encode visual attributes well (AUC > 0.86) but fail at arbitration, not perception -- analogous to decision-making failures.

### 1.4 Neuron Attribution and Representation Probing
- **NAM (NeurIPS 2024)** [11 citations]: First neuron attribution method for MLLMs, revealing cross-modal invariance and semantic sensitivity of neurons.
- **Cross-Modal Projection (ACL 2024)** [22 citations]: Found that cross-modal projection does NOT project visual attributes to textual space; domain-specific visual attributes are modeled by the LLM itself.
- **MIP-Editor (AAAI 2025)** [2 citations]: Modality-specific attribution scores to identify influential neuron paths for multimodal unlearning.
- **Structural Graph Probing (2026)** [0 citations]: Used neuron-neuron co-activation correlation graphs, found cross-modal structure consolidates with depth around recurrent hub neurons.
- **Devils in Middle Layers (CVPR 2025)** [107 citations]: Identified middle layers as crucial for visual information processing, divided into "visual information enrichment" and "semantic refinement" stages.

### 1.5 Hallucination Mechanisms
- Multiple papers converge on: hallucinations arise from progressive weakening of attention to visual tokens in deeper layers (PAINT 2025, CAAC 2025), attention distribution bias toward background tokens (DAMRO), and failures in cross-modal information aggregation at middle layers (Devils in Middle Layers).

## 2. Neuroscience of Multimodal Processing (Track B)

### 2.1 Brain Encoding/Decoding
- **Tang et al. (NeurIPS 2023)** [62 citations]: Multimodal transformers can transfer encoding across fMRI responses to stories and movies, revealing shared semantic dimensions for language and vision.
- **UMBRAE (ECCV 2024)** [40 citations]: Unified multimodal brain decoding with cross-subject training strategy.
- **Modality-Agnostic fMRI Decoding (arXiv 2024)** [5 citations]: Discovered that high-level visual (temporal) brain regions perform well on BOTH image and text stimuli, suggesting modality-agnostic representation in brain.
- **Oota et al. (ICLR 2025)** [6 citations]: Showed MLLMs have significantly better brain alignment than vision-only models; instruction-tuned MLLMs capture count-related and recognition-related concepts aligned with brain activity.
- **Monkey See, Model Knew (bioRxiv 2025)** [6 citations]: Language model predictivity of visual cortex is NOT due to language per se but to statistical structure of visual world reflected in language.

### 2.2 Brain-Model Alignment
- **Nature Communications 2025**: Brain's temporal responses to speech closely follow layer-by-layer progression of LLMs, revealing shared computational principles.
- **Lei et al. (AAAI 2025)** [2 citations]: Improvements in LLM performance drive representational architectures toward brain-like hierarchies, especially at higher semantic abstraction levels.
- **HFTP (arXiv 2025)** [3 citations]: GPT-2, Gemma, Llama models process syntax in analogous layers, while human brain relies on distinct cortical regions for different syntactic levels.
- **Soni et al. (bioRxiv 2024)** [19 citations]: Choice of similarity measure (RSA, CKA, Linear Predictivity) significantly impacts DNN-brain alignment conclusions.

### 2.3 Binding Problem
- **Campbell et al. (NeurIPS 2024)** [69 citations]: Many VLM failures on multi-object reasoning (counting, localization) can be explained by the binding problem; failure modes strikingly similar to rapid feedforward processing limitations in human brain.

### 2.4 Multimodal Integration in Brain
- **Lin et al. (Brain and Language 2024)**: Neural correlates of six semantic dimensions overlap in default mode network (left anterior superior temporal gyrus and angular gyrus), acting as connector hubs.
- **Karthik et al. (bioRxiv 2023)**: Auditory cortex encodes lipreading information through spatially distributed activity, supporting predictive mechanisms.
- **Li et al. (bioRxiv 2025)**: "Predictive vision-language integration in the human visual cortex" -- direct evidence of predictive coding in multimodal brain areas.

## 3. Bridging NeuroAI (Track C)

### 3.1 DNN-Brain Comparison Methods
- RSA, CKA, Linear Predictivity, Brain-Score
- Serious methodological concerns: Dujmovic et al. (2023) showed "mimic effect" and "modulation effect" leading to incorrect inferences about mechanism similarity.
- Soni et al. (2024) showed conclusions about DNN-brain alignment are "profoundly impacted" by similarity measure choice.

### 3.2 Brain-Aligned Visual Features
- **Brain2GAN (bioRxiv 2024)** [14 citations]: Feature-disentangled GAN w-latents outperform z-latents and CLIP in explaining neural responses.
- **Gaslight, Gatekeep, V1-V3 (2026)** [0 citations]: Brain alignment specifically in early visual cortex (V1-V3) is a reliable negative predictor of VLM sycophancy.
- **100 Neural Networks (bioRxiv 2025)** [10 citations]: Large-scale comparison of neural networks watching videos with brain activity.
- **Subramaniam et al. (ICML 2024)** [19 citations]: Used multimodal DNNs to identify sites of multimodal integration in the brain via SEEG, found CLIP-style training best predicts neural activity at integration sites.

## Timeline of Key Milestones

- **2023**: Causal tracing adapted to VLMs (BLIP); first brain encoding transfer across modalities
- **2024**: Cross-modal information flow studied in LLaVA; visual attention sinks discovered; binding problem connected to VLM failures; neuron attribution for MLLMs; brain encoding with MLLMs
- **2025**: Three-stage processing model established; circuit analysis emerges; MLLMs shown to have brain alignment; VLM arbitration failures identified
- **2026**: Full circuit tracing in VLMs; SAE-based visual thought circuits; dual-pathway hallucination circuits; structural graph probing
