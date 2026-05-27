# Gap Analysis: MLLM Interpretability from a Neuroscience Perspective

## Gap 1: No Systematic Neuroscience-Grounded Framework for MLLM Cross-Modal Processing

**What exists**: Multiple papers have independently identified that MLLMs process visual-linguistic information in stages (Zhang et al. CVPR 2025 found two stages; VisiPruner EMNLP 2025 found three stages; Devils in Middle Layers CVPR 2025 found "enrichment" and "refinement" stages). These findings are strikingly reminiscent of the ventral visual stream hierarchy in neuroscience (V1->V2->V4->IT for increasingly abstract visual processing) and the hierarchical predictive coding framework. However, NO paper has systematically mapped MLLM layer-wise processing to neuroscience-established brain processing stages. The descriptions remain purely computational/engineering.

**Specific evidence**: Zhang et al. (2024) describe their findings as "lower layers" and "middle layers" without any neuroscience grounding. VisiPruner describes "shallow layers recognize task intent" but never connects this to known brain mechanisms. The rich neuroscience literature on ventral/dorsal visual streams, hierarchical temporal processing, and predictive coding remains completely disconnected from MLLM interpretability work.

## Gap 2: No Causal Circuit Analysis Connecting MLLM Circuits to Known Brain Circuits

**What exists**: Circuit tracing has been applied to VLMs (Yang et al. 2026; Dual-Pathway Circuits 2026; Sparse Visual Thought Circuits 2026), and brain encoding/decoding studies compare DNN representations to brain representations (Tang et al. NeurIPS 2023; Oota et al. ICLR 2025). However, these two lines of work remain disconnected. No one has taken the specific circuits discovered in VLMs and asked: "Does this circuit have a functional analog in the brain?"

**Specific evidence**: The dual-pathway circuit for hallucination (grounding pathway vs. hallucination pathway, Liu et al. 2026) has a natural brain analog in the ventral "what" stream vs. dorsal "where" stream, but this connection is never made. The "arbitration failure" finding (Nooralahzadeh et al. 2026) -- where VLMs see correctly but fail to act on visual evidence -- parallels the neuroscience literature on executive function and prefrontal control, but this is not explored.

## Gap 3: The Binding Problem Has Been Connected to VLM Failures but Not to Internal Mechanisms

**What exists**: Campbell et al. (NeurIPS 2024, 69 citations) elegantly showed that many VLM failures mirror the binding problem in neuroscience. However, their analysis is purely behavioral (input-output level). They do NOT investigate the internal mechanisms: Which layers/heads/neurons are responsible for "binding" in MLLMs? Is there something analogous to the temporal binding mechanism (oscillatory synchrony) that neuroscience proposes?

**Specific evidence**: Campbell et al. explicitly state their analysis uses "behavioral" comparisons and call for future work on internal mechanisms. Meanwhile, the attention sink literature (Visual Attention Sink, ICLR 2025) shows that certain tokens receive disproportionate attention, potentially serving as binding anchors, but this connection is never drawn.

## Gap 4: No Study Applies SAEs to MLLMs to Discover Multimodal Features and Compare with Brain Features

**What exists**: SAEs have been successfully applied to LLMs (Cunningham et al. 2023, 1127 citations; Templeton et al. 2024, 442 citations) and very recently to VLMs (Sparse Visual Thought Circuits 2026), but only for task-selective feature analysis. On the brain side, RSA and CKA are used to compare DNN and brain representations (Soni et al. 2024; Dujmovic et al. 2023). However, NO one has trained SAEs on MLLMs specifically to discover multimodal integration features (features that activate for both visual and linguistic content of the same concept) and compared these to brain multimodal integration regions.

**Specific evidence**: The "cross-modal invariance" property discovered by NAM (NeurIPS 2024) suggests such features exist but uses a different methodology. Subramaniam et al. (ICML 2024) identified brain sites of multimodal integration using SEEG but used model-level (not feature-level) analysis.

## Gap 5: MLLM-to-Brain Alignment Studies Focus on Unimodal (Vision or Language) but NOT Multimodal Integration

**What exists**: Brain encoding studies compare DNN representations to brain activity (Tang et al. 2023; Oota et al. 2025; Monkey See 2025). However, these studies either: (a) use vision-only representations, (b) use language-only representations, or (c) use multimodal representations but compare against unimodal brain regions. NO study specifically targets multimodal integration regions in the brain (superior temporal sulcus, angular gyrus, prefrontal cortex) and compares them to the cross-modal processing stages identified in MLLMs.

**Specific evidence**: Nikolaus et al. (2024) found that high-level temporal regions decode both image and text stimuli, suggesting modality-agnostic representation. But they used multimodal model representations as a whole, not layer-specific representations from the multimodal processing stages. The three-stage processing model of MLLMs (VisiPruner) has never been compared to the temporal dynamics of multimodal integration in the brain.

## Gap 6: No Neuroscience-Informed Diagnostic for MLLM Failures

**What exists**: Hallucination detection methods (DAMRO, PAINT, CAAC) use engineering heuristics (attention weight thresholds, contrastive decoding). The neuroscience literature offers rich theory about why multimodal integration fails (binding problem, attentional blink, change blindness, inattentional blindness). These theories have NOT been applied as diagnostic tools for MLLM failures.

**Specific evidence**: The "arbitration failure" finding (2026) is the closest: VLMs perceive correctly but fail to act on perception. This parallels the neuroscience concept of "executive dysfunction" where perception is intact but decision-making fails. However, no one has systematically mapped MLLM failure modes to specific neuroscience failure modes.

## Gap 7: Missing Temporal Dynamics Analysis of Cross-Modal Processing in MLLMs

**What exists**: In neuroscience, the timing of multimodal integration is well-studied (multisensory response enhancement at ~100-300ms; lexical-semantic access at ~400ms N400; language-mediated prediction at ~200ms). In MLLMs, the "time" dimension is replaced by "layer" dimension. While several papers analyze layer-wise processing (Zhang et al. 2024; VisiPruner 2025; How MLLMs Solve Image Tasks 2025), none have drawn explicit parallels to the temporal dynamics of brain processing, even though the layer-to-time mapping is a natural and well-established comparison axis (Nature Communications 2025 showed brain's temporal responses follow LLM layer hierarchy).

## Summary Table

| Gap | Track A (MLLM Interp) | Track B (Neuroscience) | Track C (Bridge) |
|-----|----------------------|----------------------|------------------|
| 1. No neuro-grounded framework for MLLM stages | Multiple papers describe stages | Rich hierarchy theories exist | NOT CONNECTED |
| 2. No circuit-to-circuit comparison | VLM circuits discovered | Brain circuits well-known | NOT CONNECTED |
| 3. Binding problem: behavioral only | Internal mechanisms unknown | Binding well-studied in brain | BEHAVIORAL ONLY |
| 4. No SAE multimodal features vs. brain | SAEs just starting on VLMs | RSA/CKA exists for comparison | NOT DONE |
| 5. Alignment ignores multimodal integration | Stage model exists | Integration regions known | MISMATCHED |
| 6. No neuro-informed diagnostics | Engineering heuristics only | Rich failure theory exists | NOT APPLIED |
| 7. No temporal dynamics comparison | Layer analysis exists | Temporal dynamics well-studied | NOT MAPPED |
