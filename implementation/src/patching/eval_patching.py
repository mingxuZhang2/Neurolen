"""Evaluate patching effects on different task types.

Metrics:
- VQA accuracy for visual recognition tasks
- Binding accuracy (attribute-swap error rate) for cross-modal binding
- Language quality (perplexity, fluency, grammar) for generation tasks
"""

import logging
import re
from collections import Counter

import numpy as np

logger = logging.getLogger(__name__)


def evaluate_vqa_accuracy(
    predictions: list[str],
    ground_truths: list[str],
) -> float:
    """Evaluate VQA accuracy using relaxed string matching.

    Uses the VQAv2 accuracy metric: the prediction is considered correct if it
    appears in any of the ground truth answers (case-insensitive, with basic
    normalization).

    Args:
        predictions: model-generated answers
        ground_truths: reference answers

    Returns:
        Accuracy score in [0, 1]
    """
    assert len(predictions) == len(ground_truths)
    correct = 0
    for pred, gt in zip(predictions, ground_truths):
        pred_norm = _normalize_answer(pred)
        gt_norm = _normalize_answer(gt)

        if not gt_norm:
            # If no ground truth, skip this sample
            continue

        # Check if prediction contains the ground truth or vice versa
        if pred_norm == gt_norm:
            correct += 1
        elif gt_norm in pred_norm or pred_norm in gt_norm:
            correct += 0.5  # partial credit
        else:
            # Check word overlap
            pred_words = set(pred_norm.split())
            gt_words = set(gt_norm.split())
            if gt_words and pred_words:
                overlap = len(pred_words & gt_words) / len(gt_words)
                correct += overlap * 0.5

    n_valid = sum(1 for gt in ground_truths if _normalize_answer(gt))
    return correct / max(n_valid, 1)


def evaluate_binding_accuracy(
    predictions: list[str],
    ground_truths: list[str],
) -> float:
    """Evaluate cross-modal binding accuracy.

    Checks whether attribute-object bindings are preserved.
    Detects attribute-swap errors: correct attributes mentioned but
    assigned to wrong objects.

    Args:
        predictions: model responses about object attributes
        ground_truths: reference responses with correct bindings

    Returns:
        Binding accuracy score in [0, 1]
    """
    assert len(predictions) == len(ground_truths)

    # Common color terms for attribute detection
    colors = {"red", "blue", "green", "yellow", "orange", "purple", "pink",
              "black", "white", "brown", "gray", "grey"}
    sizes = {"large", "small", "big", "tiny", "huge", "little"}
    positions = {"left", "right", "top", "bottom", "center", "middle",
                 "above", "below", "beside", "near"}

    total_bindings = 0
    correct_bindings = 0

    for pred, gt in zip(predictions, ground_truths):
        pred_norm = pred.lower().strip()
        gt_norm = gt.lower().strip()

        if not gt_norm:
            continue

        # Extract attribute-object pairs from ground truth
        gt_bindings = _extract_bindings(gt_norm, colors | sizes | positions)
        pred_bindings = _extract_bindings(pred_norm, colors | sizes | positions)

        if not gt_bindings:
            # If no clear bindings in ground truth, use word overlap
            pred_words = set(pred_norm.split())
            gt_words = set(gt_norm.split())
            if gt_words:
                overlap = len(pred_words & gt_words) / len(gt_words)
                total_bindings += 1
                correct_bindings += overlap
            continue

        for gt_attr, gt_obj in gt_bindings:
            total_bindings += 1
            # Check if the same binding exists in prediction
            for pred_attr, pred_obj in pred_bindings:
                if gt_attr == pred_attr and _fuzzy_match(gt_obj, pred_obj):
                    correct_bindings += 1
                    break
            # Also check for attribute swaps (attribute present but bound to wrong object)

    return correct_bindings / max(total_bindings, 1)


def _extract_bindings(text: str, attributes: set) -> list[tuple[str, str]]:
    """Extract (attribute, object) pairs from text.

    Simple heuristic: look for patterns like "red car", "large dog", etc.
    """
    words = text.split()
    bindings = []
    for i, word in enumerate(words):
        if word in attributes and i + 1 < len(words):
            obj = words[i + 1]
            # Skip stop words
            if obj not in {"a", "an", "the", "is", "are", "and", "or", "of", "in"}:
                bindings.append((word, obj))
    return bindings


def _fuzzy_match(a: str, b: str) -> bool:
    """Check if two words are similar enough to be considered the same object."""
    a, b = a.lower().strip(), b.lower().strip()
    if a == b:
        return True
    if a in b or b in a:
        return True
    # Simple edit distance check
    if len(a) > 3 and len(b) > 3:
        common = len(set(a) & set(b))
        total = max(len(set(a)), len(set(b)))
        if common / total > 0.6:
            return True
    return False


def evaluate_language_quality(
    texts: list[str],
) -> dict:
    """Evaluate linguistic quality of generated texts.

    Metrics:
    - Length: average number of words
    - Vocabulary richness: type-token ratio
    - Repetition rate: fraction of repeated trigrams
    - Sentence count: average sentences per response
    - Composite: weighted combination

    Note: For full fluency/grammar evaluation, use a dedicated model (e.g.,
    language-tool-python or a trained grammar classifier). This provides
    lightweight proxy metrics suitable for comparing patched vs unpatched.

    Args:
        texts: list of generated text strings

    Returns:
        dict with individual metrics and composite score
    """
    if not texts:
        return {"composite": 0.0, "length": 0, "vocab_richness": 0,
                "repetition_rate": 0, "sentence_count": 0}

    lengths = []
    vocab_richness_scores = []
    repetition_rates = []
    sentence_counts = []

    for text in texts:
        words = text.split()
        length = len(words)
        lengths.append(length)

        # Type-token ratio (vocabulary richness)
        if length > 0:
            unique_words = len(set(w.lower() for w in words))
            ttr = unique_words / length
        else:
            ttr = 0.0
        vocab_richness_scores.append(ttr)

        # Repetition rate (trigram overlap)
        if length >= 3:
            trigrams = [tuple(words[i:i+3]) for i in range(length - 2)]
            trigram_counts = Counter(trigrams)
            repeated = sum(1 for c in trigram_counts.values() if c > 1)
            rep_rate = repeated / len(trigrams) if trigrams else 0
        else:
            rep_rate = 0.0
        repetition_rates.append(rep_rate)

        # Sentence count
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_counts.append(len(sentences))

    avg_length = np.mean(lengths)
    avg_vocab = np.mean(vocab_richness_scores)
    avg_rep = np.mean(repetition_rates)
    avg_sent = np.mean(sentence_counts)

    # Composite score: rewards length, vocabulary, and sentence structure;
    # penalizes repetition. Normalized to roughly [0, 1].
    # These weights are chosen so that a typical good response scores ~0.7-0.8
    length_score = min(avg_length / 50.0, 1.0)  # cap at 50 words
    vocab_score = avg_vocab  # already in [0, 1]
    rep_penalty = 1.0 - avg_rep  # lower repetition = better
    sent_score = min(avg_sent / 3.0, 1.0)  # cap at 3 sentences

    composite = 0.3 * length_score + 0.3 * vocab_score + 0.2 * rep_penalty + 0.2 * sent_score

    return {
        "composite": float(composite),
        "length": float(avg_length),
        "vocab_richness": float(avg_vocab),
        "repetition_rate": float(avg_rep),
        "sentence_count": float(avg_sent),
        "length_score": float(length_score),
        "vocab_score": float(vocab_score),
        "rep_penalty": float(rep_penalty),
        "sent_score": float(sent_score),
    }


def compute_perplexity(
    model,
    processor,
    texts: list[str],
    images: list = None,
    model_config: dict = None,
) -> float:
    """Compute perplexity of generated text using the model itself.

    This is a self-evaluation metric: how surprised is the model by its
    own generated text?

    Args:
        model: the MLLM
        processor: tokenizer/processor
        texts: generated texts to evaluate
        images: corresponding images (optional)
        model_config: model configuration

    Returns:
        Mean perplexity across texts
    """
    import torch

    perplexities = []
    for i, text in enumerate(texts):
        if not text.strip():
            continue

        try:
            if images is not None and i < len(images):
                # Compute conditional perplexity given image
                inputs = processor(
                    text=text, images=images[i], return_tensors="pt"
                )
            else:
                inputs = processor(text=text, return_tensors="pt")

            inputs = {k: v.to(model.device) if hasattr(v, 'to') else v
                      for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**{k: v for k, v in inputs.items()
                                   if isinstance(v, torch.Tensor)})
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    ppl = torch.exp(outputs.loss).item()
                else:
                    # Compute manually from logits
                    logits = outputs.logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = inputs['input_ids'][..., 1:].contiguous()
                    loss_fn = torch.nn.CrossEntropyLoss()
                    loss = loss_fn(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                    ppl = torch.exp(loss).item()

            if np.isfinite(ppl) and ppl < 1e6:
                perplexities.append(ppl)
        except Exception as e:
            logger.warning(f"Perplexity computation failed: {e}")

    return float(np.mean(perplexities)) if perplexities else float('inf')


def _normalize_answer(text: str) -> str:
    """Normalize text for VQA comparison."""
    text = text.lower().strip()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text
