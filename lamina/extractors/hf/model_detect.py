"""
Model-type detection for HuggingFace models.

``_model_can_generate(model)`` decides whether to run ``model.generate()``
(causal LM, seq2seq LM, conditional generation) or a plain forward pass
(encoder-only, classifiers, QA, masked LM, …).
"""
from __future__ import annotations

from typing import Tuple

# ── Suffix tables ─────────────────────────────────────────────────────────────

# Class-name substrings → generative (use generate())
_GENERATIVE_SUFFIXES: Tuple[str, ...] = (
    "ForCausalLM",
    "LMHeadModel",               # GPT-2, GPT-J, …
    "ForSeq2SeqLM",
    "ForConditionalGeneration",  # T5, BART, mT5, mBART, …
    "ForSpeechSeq2Seq",
    "ForVision2Seq",
    "ForTextGeneration",
    "ForTextToWaveform",
    "ForTextToSpectrogram",
)

# Class-name substrings → non-generative (use forward())
_NON_GENERATIVE_SUFFIXES: Tuple[str, ...] = (
    "ForMaskedLM",
    "ForSequenceClassification",
    "ForTokenClassification",
    "ForQuestionAnswering",
    "ForMultipleChoice",
    "ForNextSentencePrediction",
    "ForPreTraining",
    "ForImageClassification",
    "ForImageSegmentation",
    "ForSemanticSegmentation",
    "ForPanopticSegmentation",
    "ForObjectDetection",
    "ForDepthEstimation",
    "ForAudioClassification",
    "ForAudioFrameClassification",
    "ForAudioXVector",
    "ForCTC",
    "ForFeatureExtraction",
    "ForZeroShotClassification",
    "ForZeroShotImageClassification",
    "ForZeroShotObjectDetection",
    "ForSentenceSimilarity",
)

# Well-known encoder-only model_type strings from HF config
_ENCODER_ONLY_MODEL_TYPES: frozenset = frozenset({
    "bert", "roberta", "albert", "distilbert", "deberta", "deberta-v2",
    "electra", "xlm-roberta", "camembert", "xlm", "flaubert", "rembert",
    "big_bird", "longformer", "roformer",
    "layoutlm", "layoutlmv2", "layoutlmv3", "lilt",
    "visual_bert", "clip_text_model",
})


def _model_can_generate(model) -> bool:
    """
    Return ``True`` if the model should be run via ``model.generate()``,
    ``False`` if it should use a plain ``model(**inputs)`` forward call.

    Resolution order
    ----------------
    1. ``model.can_generate()`` — HF's official API (transformers ≥ 4.18).
    2. Class-name suffix matched against ``_GENERATIVE_SUFFIXES`` /
       ``_NON_GENERATIVE_SUFFIXES``.
    3. ``model.config.is_encoder_decoder`` — enc-dec models always generate.
    4. ``model.config.model_type`` matched against ``_ENCODER_ONLY_MODEL_TYPES``.
    5. Default: assume generative.
    """
    # 1. HF's own authoritative API
    can_gen = getattr(model, "can_generate", None)
    if callable(can_gen):
        try:
            return bool(can_gen())
        except Exception:
            pass

    cls_name = type(model).__name__

    # 2. Class-name suffix heuristics
    if any(suf in cls_name for suf in _GENERATIVE_SUFFIXES):
        return True
    if any(suf in cls_name for suf in _NON_GENERATIVE_SUFFIXES):
        return False

    # 3/4. Config-level checks
    cfg = getattr(model, "config", None)
    if cfg is not None:
        if getattr(cfg, "is_encoder_decoder", False):
            return True
        if getattr(cfg, "model_type", "") in _ENCODER_ONLY_MODEL_TYPES:
            return False

    # 5. Default: assume generative
    return True
