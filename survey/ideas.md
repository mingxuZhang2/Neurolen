# Research Ideas: Neuroscience-Grounded Interpretability of MLLMs

## Idea 1: NeuroLens -- Mapping MLLM Cross-Modal Processing Stages to Brain Functional Hierarchy

**One-line summary**: Systematically map the layer-wise cross-modal processing stages of MLLMs (visual grounding, cross-modal fusion, linguistic refinement) onto neuroscience-established brain processing hierarchies (ventral visual stream, multimodal integration areas, language areas) using representational similarity analysis.

**Gap addressed**: Gaps 1, 5, and 7. Multiple papers have independently discovered that MLLMs process cross-modal information in stages (Zhang et al. CVPR 2025; VisiPruner EMNLP 2025; Devils in Middle Layers CVPR 2025), but none have grounded these findings in neuroscience. Meanwhile, brain encoding studies (Tang et al. NeurIPS 2023; Oota et al. ICLR 2025) compare model representations to brain activity without targeting the stage-specific processing.

**Core method**: (1) Extract layer-wise representations from multiple MLLMs (LLaVA-1.5, Qwen2-VL, InternVL) at each transformer layer while processing image-text pairs. (2) Segment layers into processing stages using the established three-stage model (Zhang et al.; VisiPruner). (3) Using fMRI data from publicly available datasets (NSD, BOLD5000, or the dataset from Nikolaus et al. 2024 which has paired image-text stimuli), compute representational similarity (RSA and CKA) between each MLLM stage and specific brain regions: early visual cortex (V1-V3) for the visual encoding stage, multimodal integration regions (STS, angular gyrus, TPJ) for the cross-modal fusion stage, and language areas (Broca's, Wernicke's) for the linguistic refinement stage. (4) Validate the mapping by showing that perturbing specific MLLM stages (via activation patching) produces failures that mirror known effects of damage to corresponding brain regions (e.g., perturbing the "fusion" stage should cause binding-like failures, similar to parietal lesions).

**Why it works without wet lab**: All neuroscience data come from publicly available fMRI datasets (NSD has 73,000 trials across 8 subjects; BOLD5000 has 5,254 images). All MLLM analysis is purely computational (activation extraction, RSA, activation patching). No new brain scans required.

**Feasibility**: High. Uses existing open-source MLLMs (LLaVA, Qwen2-VL available on HuggingFace), established analysis toolkits (rsatoolbox for RSA, torch-cka for CKA), and public fMRI datasets. The three-stage model is already established; this paper provides the neuroscience grounding. Code from Zhang et al. (crossmodal-information-flow-in-MLLM) is available on GitHub. Estimated 2-3 weeks for implementation, 2-3 weeks for experiments.

**Novelty assessment**: This would be the first paper to create a systematic bridge between MLLM internal processing stages and brain functional hierarchy for multimodal processing. While individual components exist (MLLM stage analysis exists, brain encoding exists), the systematic mapping with stage-specific correspondence and causal validation is novel. The closest work (Oota et al. ICLR 2025) compares MLLMs to brain at the model level, not at the stage level.

**Expected contribution**:
1. Framework contribution: A neuroscience-grounded framework (NeuroLens) for interpreting MLLM cross-modal processing, establishing a systematic mapping between MLLM processing stages and brain functional regions.
2. Component contribution: A stage-specific representational alignment method that goes beyond whole-model brain comparison by matching processing stages to brain areas.
3. Experimental contribution: Comprehensive empirical evidence across multiple MLLMs and brain datasets showing convergent cross-modal processing principles, plus causal validation through activation patching.

**Target venues**: NeurIPS 2026 (main track), ICML 2026, or Nature Machine Intelligence (if results are strong enough for a shorter format with high impact). The NeuroAI angle makes it suitable for top ML venues that have been receptive to interdisciplinary work.

**Estimated effort**: 4-5 weeks implementation + 3-4 weeks experiments = ~8 weeks total.

---

## Idea 2: Multimodal Binding Circuits -- Discovering How MLLMs Solve (or Fail) the Feature Binding Problem

**One-line summary**: Use mechanistic interpretability tools (activation patching, SAEs) to identify the specific circuits in MLLMs responsible for binding visual features to objects, and compare these to the neural mechanisms of feature binding in the brain (temporal synchrony, attentional selection).

**Gap addressed**: Gaps 2, 3, and 4. Campbell et al. (NeurIPS 2024, 69 citations) showed VLM failures mirror the binding problem but only at the behavioral level. No one has investigated the internal mechanisms of binding in MLLMs, and no one has compared MLLM binding mechanisms to brain binding mechanisms.

**Core method**: (1) Design controlled experiments with binding-demanding stimuli (e.g., "a red circle and a blue square" vs. "a blue circle and a red square") that systematically vary the binding demands. (2) Apply activation patching across all layers and attention heads of LLaVA-1.5 and Qwen2-VL to identify which components are critical for correct binding (attributing the right feature to the right object). (3) Train sparse autoencoders (SAEs) at the identified critical layers to discover binding-related features (features that activate when a specific attribute is correctly bound to a specific object). (4) Characterize the binding mechanism: Is it attention-based (specific heads that attend to both the object and its attribute)? Is it position-based (spatial co-location)? (5) Compare the discovered mechanism to three brain binding theories: (a) temporal synchrony (do binding heads show correlated activation patterns?), (b) attentional selection (does task attention modulate binding?), (c) spatial binding (is position information critical?). (6) Demonstrate that binding circuit failures predict hallucination types (attribute swap hallucinations vs. object absence hallucinations).

**Why it works without wet lab**: The binding problem can be studied computationally using synthetic multi-object stimuli. Campbell et al. already created behavioral benchmarks. Brain theories make specific predictions about computational mechanisms that can be tested in MLLMs. No brain data needed; the comparison is theory-to-mechanism.

**Feasibility**: Moderate-High. Activation patching tools exist (TransformerLens adaptable to VLMs, or use the approach from FCCT/AAAI 2025). SAE training follows established recipes (Cunningham et al. 2023). The main challenge is training SAEs on multimodal models, which has only been done once (Sparse Visual Thought Circuits 2026 on Qwen3-VL). Estimated 3-4 weeks for implementation, 3-4 weeks for experiments.

**Novelty assessment**: Highly novel. Campbell et al. explicitly call for internal mechanism analysis as future work. No existing paper combines (a) binding-specific experiments, (b) circuit-level analysis in MLLMs, and (c) comparison to neuroscience binding theories. The discovery of binding circuits would be a foundational contribution to both MLLM interpretability and computational neuroscience.

**Expected contribution**:
1. Framework contribution: First mechanistic analysis of the binding problem in MLLMs, bridging the behavioral findings of Campbell et al. with internal circuit analysis.
2. Component contribution: Discovery and characterization of "binding circuits" in MLLMs, including binding-specific attention heads and SAE features.
3. Experimental contribution: Demonstration that binding circuit failures causally predict specific hallucination types (attribute swap vs. object absence), with comparison to three neuroscience binding theories.

**Target venues**: NeurIPS 2026 (Oral-tier potential due to novelty), ICLR 2027, or Neuron/Nature Neuroscience (if the brain comparison is compelling enough for a neuroscience audience).

**Estimated effort**: 5-6 weeks implementation + 4-5 weeks experiments = ~10 weeks total.

---

## Idea 3: Cross-Modal Sparse Features -- Training SAEs on MLLMs to Discover Multimodal Integration Features

**One-line summary**: Train sparse autoencoders on the intermediate layers of MLLMs to discover interpretable features that capture cross-modal concepts, then analyze whether these features exhibit properties analogous to multimodal neurons in the brain (modality invariance, multisensory enhancement, inverse effectiveness).

**Gap addressed**: Gap 4 and partially Gap 2. SAEs have been applied to LLMs (Cunningham et al. 2023) and once to VLMs (Zhou 2026), but never with the goal of discovering and characterizing multimodal integration features. The "cross-modal invariance" property (NAM, NeurIPS 2024) suggests such features exist but was found using attribution, not SAEs.

**Core method**: (1) Select the critical multimodal integration layers identified by prior work (middle layers, per Zhang et al. 2024 and VisiPruner 2025). (2) Train SAEs at these layers on LLaVA-1.5-7B using a large activation dataset from mixed image-text inputs. (3) Classify discovered features into: (a) vision-only features (activate only for visual content), (b) language-only features, (c) cross-modal features (activate for both visual and linguistic representations of the same concept). (4) For the cross-modal features, test three neuroscience-predicted properties: modality invariance (same concept activates the feature regardless of input modality), multisensory enhancement (feature activates more strongly when both modalities are present), and inverse effectiveness (enhancement is stronger for weaker unimodal inputs). (5) Use the discovered features for interpretable analysis of hallucination: which features fire when the model hallucinates? Are hallucinations caused by failure of cross-modal features or spurious activation of unimodal features?

**Why it works without wet lab**: Entirely computational. SAE training requires GPU compute (H100s on HPC3). Feature characterization uses controlled input manipulation (unimodal vs. multimodal stimuli). The neuroscience properties (modality invariance, multisensory enhancement, inverse effectiveness) are well-defined computational properties that can be measured directly.

**Feasibility**: Moderate. SAE training on large models is computationally expensive but established. The main risk is whether clean multimodal features emerge at the scale we can train at. Zhou (2026) successfully trained SAEs on Qwen3-VL-8B, suggesting feasibility. Can leverage the SAELens library with modifications for multimodal models. Estimated 3-4 weeks for SAE training infrastructure + 4-5 weeks for analysis.

**Novelty assessment**: Novel. This would be the first SAE study specifically targeting multimodal integration features in MLLMs, and the first to test neuroscience-predicted properties (multisensory enhancement, inverse effectiveness) in SAE features. The closest work (Zhou 2026) trains SAEs on VLMs but for task-selective analysis, not multimodal feature discovery.

**Expected contribution**:
1. Framework contribution: First application of SAEs to discover and characterize multimodal integration features in MLLMs, establishing a neuroscience-inspired taxonomy of learned features.
2. Component contribution: Discovery that MLLM features exhibit (or fail to exhibit) properties analogous to multimodal neurons in the brain, providing new understanding of how multimodal representations emerge.
3. Experimental contribution: Interpretable analysis of hallucination through the lens of multimodal feature activation, showing specific failure patterns in cross-modal vs. unimodal features.

**Target venues**: ICML 2026, NeurIPS 2026, or ICLR 2027. Suitable for ML interpretability tracks.

**Estimated effort**: 5-6 weeks implementation + 5-6 weeks experiments = ~11 weeks total.

---

## Idea 4: Predictive Coding in MLLMs -- Testing Whether MLLMs Implement Brain-Like Top-Down Prediction

**One-line summary**: Test whether MLLMs implement a form of predictive coding (a dominant neuroscience theory of brain computation) by analyzing whether higher layers generate predictions that are compared against lower-layer visual inputs, and whether prediction errors drive cross-modal information integration.

**Gap addressed**: Gaps 1 and 7. Predictive coding is arguably the most influential framework in computational neuroscience for understanding perception (including multimodal perception), yet it has never been tested as a model of MLLM information processing. The Nature Communications (2025) finding that brain temporal responses follow LLM layer hierarchy strongly suggests a predictive coding interpretation is viable.

**Core method**: (1) Define testable predictions of the predictive coding framework for MLLMs: (a) representations at each layer should be decomposable into a "prediction" component (from higher layers via skip connections or attention) and a "prediction error" component; (b) unexpected visual content should produce larger activation changes at cross-modal integration layers; (c) context (text prompt) should modulate visual processing by setting predictions. (2) Measure prediction and prediction error in MLLMs by comparing activations to expected (text-congruent) vs. unexpected (text-incongruent) visual inputs, controlling for low-level features. (3) Analyze whether the three-stage processing model (VisiPruner) can be reinterpreted as: Stage 1 (prediction generation from text), Stage 2 (prediction-visual comparison at integration layers), Stage 3 (error-corrected response generation). (4) Test the "inverse effectiveness" prediction: the model's reliance on cross-modal integration should increase when unimodal signals are ambiguous (low-quality images, vague text). (5) Compare the prediction error signals in MLLMs to established neural markers of prediction error in the brain (mismatch negativity at ~100ms, N400 at ~400ms for semantic mismatch).

**Why it works without wet lab**: All analysis is computational. "Prediction error" can be measured as the difference in activations between congruent and incongruent image-text pairs. Brain comparison is against established ERP/fMRI findings from the literature, not new experiments.

**Feasibility**: Moderate. The experimental design is straightforward (congruent vs. incongruent stimuli), and the analysis tools (activation extraction, comparison) are standard. The main challenge is operationalizing "prediction" and "prediction error" in the transformer architecture, which lacks the explicit recurrent structure of predictive coding models. However, the attention mechanism (which allows higher layers to influence lower layers through residual connections) provides an analogous pathway. Estimated 3-4 weeks implementation + 3-4 weeks experiments.

**Novelty assessment**: Highly novel. While predictive coding has been discussed in relation to deep learning generally, it has never been systematically tested in MLLMs. The connection between the three-stage MLLM processing model and predictive coding is completely unexplored. This would be the first paper to test a specific neuroscience theory of computation against MLLM internal mechanisms.

**Expected contribution**:
1. Framework contribution: A predictive coding interpretation of MLLM cross-modal processing, providing a neuroscience-grounded theoretical framework for understanding why MLLMs process information in stages.
2. Component contribution: Methods for measuring prediction and prediction error signals in MLLMs, analogous to mismatch detection in the brain.
3. Experimental contribution: Evidence for or against predictive coding in MLLMs across multiple model families, with quantitative comparison to known brain prediction error signals.

**Target venues**: NeurIPS 2026 (strong NeuroAI angle), Nature Machine Intelligence (if framed as a discovery), or ICLR 2027. The theoretical depth makes it suitable for top venues.

**Estimated effort**: 4-5 weeks implementation + 4-5 weeks experiments = ~9 weeks total.

---

## Idea 5: Ventral-Dorsal Dissociation in MLLMs -- Do MLLMs Develop Separate "What" and "Where" Pathways?

**One-line summary**: Investigate whether MLLMs develop functionally distinct pathways for object identity ("what") and spatial relations ("where"), analogous to the ventral and dorsal visual streams in the brain, using probing, ablation, and circuit analysis.

**Gap addressed**: Gaps 1 and 2. The ventral/dorsal pathway distinction is one of the most fundamental organizing principles in visual neuroscience, yet it has never been tested in MLLMs. The "Dual-Pathway Circuits" paper (2026) found a grounding pathway vs. hallucination pathway, but this is a different distinction. The question is whether MLLMs develop functionally separate pathways for object recognition vs. spatial reasoning.

**Core method**: (1) Design probing tasks that selectively require either object identity (What is this? What color?) or spatial relations (Where is X relative to Y? How many objects?). (2) Train linear probes at each layer and each attention head to predict answers to identity vs. spatial questions separately. (3) Identify heads/layers that are selectively important for identity vs. spatial tasks using activation patching. (4) Test the double dissociation: ablating "identity" heads should hurt object recognition but not spatial reasoning, and vice versa. (5) If dissociation exists, compare it to the ventral/dorsal stream: Does the "identity" pathway share representational properties with the ventral stream (invariance to position/size, sensitivity to shape/texture)? Does the "spatial" pathway share properties with the dorsal stream (sensitivity to location, motion, spatial relations, but less to object identity)? (6) Use the discovered dissociation to explain and predict specific failure modes (e.g., MLLMs are known to be worse at spatial reasoning than object recognition, paralleling the relative immaturity of dorsal stream processing in rapid feedforward passes).

**Why it works without wet lab**: Entirely computational. Uses controlled stimuli to probe identity vs. spatial processing. The brain comparison is against well-established properties of ventral/dorsal streams from decades of neuroscience research.

**Feasibility**: High. Linear probing and activation patching are well-established. The main contribution is the experimental design (identity vs. spatial tasks) and the neuroscience framing. Can use existing VQA datasets annotated for spatial reasoning (GQA, VSR) alongside standard object recognition benchmarks. Estimated 3-4 weeks implementation + 3-4 weeks experiments.

**Novelty assessment**: Novel. While the ventral/dorsal distinction is textbook neuroscience, it has never been tested as an organizational principle in MLLMs. The closest work is Campbell et al. (2024) on binding, which is related but different (binding requires combining "what" and "where"). This would be a clean, interpretable study with strong neuroscience grounding.

**Expected contribution**:
1. Framework contribution: First investigation of whether MLLMs develop ventral/dorsal-like functional dissociation, establishing a neuroscience-inspired lens for understanding MLLM spatial reasoning limitations.
2. Component contribution: Discovery and characterization of "identity heads" and "spatial heads" in MLLMs, providing interpretable units for analyzing multimodal processing.
3. Experimental contribution: Systematic probing and ablation experiments across multiple MLLMs, demonstrating the presence (or absence) of double dissociation, with comparison to known brain pathway properties.

**Target venues**: NeurIPS 2026, CVPR 2026 (visual neuroscience angle), or Cognition/Trends in Cognitive Sciences (if framed as a cognitive science contribution).

**Estimated effort**: 3-4 weeks implementation + 3-4 weeks experiments = ~7 weeks total.
