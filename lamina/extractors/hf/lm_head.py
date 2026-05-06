"""
LM-head and final-norm discovery utilities for logit-lens extraction.

``register_model(model)`` is the only public function; it walks common
attribute paths to find the LM-head weight matrix and the final LayerNorm,
copies them to CPU numpy, and stores them in the extractor module's globals
so the background worker can compute logit-lens projections without GPU
access.

Supported architectures
-----------------------
GPT-2, GPT-J, Falcon, GPT-NeoX/Pythia, LLaMA/Mistral/Gemma, OPT, BLOOM,
and anything that follows one of the well-known attribute paths below.
"""
from __future__ import annotations

import warnings
from typing import Any, Callable, Optional, Tuple

import numpy as np


# ── Common attribute paths ────────────────────────────────────────────────────

_FINAL_NORM_PATHS: Tuple[str, ...] = (
    "model.norm",                            # LLaMA, Mistral, Gemma
    "transformer.ln_f",                      # GPT-2, Falcon, GPT-J
    "gpt_neox.final_layer_norm",             # GPT-NeoX / Pythia
    "model.decoder.final_layer_norm",        # OPT
    "transformer.word_embeddings_layernorm", # BLOOM (after blocks)
    "model.final_layernorm",                 # Persimmon / some Falcons
)

_LM_HEAD_PATHS: Tuple[str, ...] = (
    "lm_head",
    "embed_out",   # GPT-NeoX
)


def _getattr_chain(obj: Any, dotted: str) -> Optional[Any]:
    """Resolve a dotted attribute path, returning None if any step fails."""
    for part in dotted.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def find_lm_head(
    model: Any,
) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
    """
    Locate the LM-head linear layer and return (weight, bias) as numpy arrays.

    Returns ``None`` if no LM-head is found or if it has no weight attribute.
    """
    lm_head = None
    for path in _LM_HEAD_PATHS:
        lm_head = _getattr_chain(model, path)
        if lm_head is not None:
            break

    if lm_head is None:
        return None

    weight = getattr(lm_head, "weight", None)
    if weight is None:
        return None

    w_arr = weight.detach().cpu().float().numpy()
    bias   = getattr(lm_head, "bias", None)
    b_arr  = bias.detach().cpu().float().numpy() if bias is not None else None
    return w_arr, b_arr


def find_final_norm(model: Any) -> Callable[[np.ndarray], np.ndarray]:
    """
    Locate the final LayerNorm/RMSNorm and return a CPU-numpy callable.

    Falls back to an identity function if no norm is found, with a warning.
    """
    norm_module = None
    for path in _FINAL_NORM_PATHS:
        norm_module = _getattr_chain(model, path)
        if norm_module is not None:
            break

    if norm_module is None:
        warnings.warn(
            "[lamina] Could not find the final LayerNorm. "
            "Logit-lens projections will use unnormalized hidden states.",
            stacklevel=3,
        )
        return lambda x: x

    norm_weight = getattr(norm_module, "weight", None)
    norm_bias   = getattr(norm_module, "bias", None)
    eps         = getattr(norm_module, "eps", 1e-5)

    nw = norm_weight.detach().cpu().float().numpy() if norm_weight is not None else None
    nb = norm_bias.detach().cpu().float().numpy()   if norm_bias   is not None else None

    def _norm_fn(
        x: np.ndarray,
        _nw=nw, _nb=nb, _eps=eps,
    ) -> np.ndarray:
        x = x.astype(np.float32)
        if _nw is not None and _nb is not None:
            # LayerNorm
            mean = x.mean(axis=-1, keepdims=True)
            var  = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
            x = (x - mean) / np.sqrt(var + _eps)
            x = x * _nw + _nb
        elif _nw is not None:
            # RMSNorm (LLaMA-style)
            rms = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + _eps)
            x = x / rms * _nw
        return x

    return _norm_fn


def register_model(model: Any) -> None:
    """
    Copy LM-head weights and final LayerNorm parameters to CPU numpy so the
    background thread can compute logit-lens projections without GPU access.

    Call this once after loading the model::

        import lamina
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        lamina.register_model(model)

    If the LM-head or LayerNorm cannot be found, a warning is emitted and
    logit-lens extraction is silently disabled for this model.
    """
    # Import here to avoid circular dependency at module load time
    from . import extractor as _ext

    result = find_lm_head(model)
    if result is None:
        warnings.warn(
            "[lamina] Could not find lm_head on the model. "
            "Logit-lens extraction will be disabled.",
            stacklevel=2,
        )
        return

    _ext._lm_head_weight, _ext._lm_head_bias = result
    _ext._final_norm_fn = find_final_norm(model)
