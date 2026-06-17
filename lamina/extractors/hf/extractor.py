"""
HuggingFace Transformers extractor.

Installs two interception points:

1. ``GenerationMixin.generate`` — wraps every ``model.generate()`` call and
   registers a ``register_forward_hook`` on the model instance so every
   forward pass within generate is captured.

2. ``run_forward()`` — for non-generative models (encoder-only, classifiers,
   QA, MLM), call this instead of ``model(**inputs)`` directly.  It registers
   the same hook for the duration of the single forward pass.

Note on ``_patched_forward``
----------------------------
A patch on ``PreTrainedModel.forward`` is also installed as a last-resort
fallback, but it will never fire for real HuggingFace models because every
concrete subclass defines its own ``forward()`` override (Python MRO finds
it first).  The correct path for encoder-only models is ``run_forward()``.
"""
from __future__ import annotations

import uuid
import warnings
from typing import Any, Callable, Dict, Optional

import numpy as np

from lamina.core.config import InternalsConfig
from lamina.core.store import InternalsStore
from lamina.core.worker import BackgroundWorker

# ---------------------------------------------------------------------------
# Module-level state (populated by _initialise())
# ---------------------------------------------------------------------------

_config: Optional[InternalsConfig] = None
_store:  Optional[InternalsStore]  = None
_worker: Optional[BackgroundWorker] = None

# Logit-lens artefacts (set by lm_head.register_model())
_lm_head_weight: Optional[np.ndarray] = None
_lm_head_bias:   Optional[np.ndarray] = None
_final_norm_fn:  Optional[Callable]   = None

# Last run started — lets callers retrieve the run after generate() returns
_last_started_run_id: Optional[str] = None

# Per-run thinking-end token ID set by dataset.py before each generate() call.
# Reset to None immediately after the run starts so it doesn't bleed into
# the next run.
_next_thinking_end_token_id: Optional[int] = None


def _initialise(
    config: InternalsConfig,
    store: InternalsStore,
    worker: BackgroundWorker,
) -> None:
    global _config, _store, _worker
    _config = config
    _store  = store
    _worker = worker


# ---------------------------------------------------------------------------
# Forward hook factory
# ---------------------------------------------------------------------------

def _make_forward_hook(
    run_id: str,
    config: InternalsConfig,
    worker: BackgroundWorker,
    step_counter: list,   # mutable int box
) -> Callable:

    def _hook(module, inputs, output) -> None:
        item: Dict[str, Any] = {
            "kind":     "step",
            "run_id":   run_id,
            "step_idx": step_counter[0],
            "config":   config,
        }

        if config.extract_hidden_states:
            # decoder_hidden_states for encoder-decoder models (T5, BART)
            hs = getattr(output, "hidden_states", None)
            if hs is None:
                hs = getattr(output, "decoder_hidden_states", None)
            item["hidden_states"] = hs
            item["encoder_hidden_states"] = getattr(
                output, "encoder_hidden_states", None
            )

        if config.extract_attentions:
            att = getattr(output, "attentions", None)
            if att is None:
                att = getattr(output, "decoder_attentions", None)
            item["attentions"] = att
            item["encoder_attentions"] = getattr(
                output, "encoder_attentions", None
            )

        if config.extract_logits:
            logits = getattr(output, "logits", None)
            item["logits"] = logits
            if logits is None:
                # QA models expose start_logits / end_logits
                item["start_logits"] = getattr(output, "start_logits", None)
                item["end_logits"]   = getattr(output, "end_logits",   None)
            else:
                item["start_logits"] = None
                item["end_logits"]   = None

            # Signal how to interpret the tensor downstream
            out_type = type(output).__name__
            item["logit_mode"] = "last_token" if any(
                tag in out_type for tag in (
                    "CausalLM", "Seq2SeqLM", "ConditionalGeneration",
                    "SpeechSeq2Seq", "Vision2Seq",
                )
            ) else "full"

        worker.enqueue_step(item)
        step_counter[0] += 1

    return _hook


def _end_payload(config: InternalsConfig, run_id: str) -> dict:
    """Build the 'end' queue item, including logit-lens artefacts if enabled."""
    return {
        "kind":           "end",
        "run_id":         run_id,
        "lm_head_weight": _lm_head_weight if config.extract_logit_lens else None,
        "lm_head_bias":   _lm_head_bias   if config.extract_logit_lens else None,
        "final_norm_fn":  _final_norm_fn  if config.extract_logit_lens else None,
    }


# ---------------------------------------------------------------------------
# Patched generate
# ---------------------------------------------------------------------------

_ORIGINAL_GENERATE: Optional[Callable] = None


def _patched_generate(self, input_ids, **kwargs):
    if _config is None or _store is None or _worker is None:
        return _ORIGINAL_GENERATE(self, input_ids, **kwargs)

    config = _config

    if config.extract_hidden_states:
        kwargs.setdefault("output_hidden_states", True)
    if config.extract_attentions:
        kwargs.setdefault("output_attentions", True)
    kwargs.setdefault("return_dict_in_generate", True)

    global _last_started_run_id, _next_thinking_end_token_id
    run_id    = str(uuid.uuid4())
    input_len = input_ids.shape[-1]
    thinking_token_id = _next_thinking_end_token_id
    _next_thinking_end_token_id = None   # consume — don't bleed into next run
    _store.start_run(
        run_id,
        input_len,
        config=config,
        thinking_end_token_id=thinking_token_id,
    )
    _last_started_run_id = run_id

    step_counter = [0]
    hook_handle  = self.register_forward_hook(
        _make_forward_hook(run_id, config, _worker, step_counter)
    )

    try:
        result = _ORIGINAL_GENERATE(self, input_ids, **kwargs)
    finally:
        hook_handle.remove()
        _worker.enqueue_end(_end_payload(config, run_id))

    return result


# ---------------------------------------------------------------------------
# Patched forward (fallback — rarely fires for real HF models; see module doc)
# ---------------------------------------------------------------------------

_ORIGINAL_FORWARD: Optional[Callable] = None
_FORWARD_RUN_KEY = "__lamina_run_id__"


def _patched_forward(self, *args, **kwargs):
    if _config is None or _store is None or _worker is None:
        return _ORIGINAL_FORWARD(self, *args, **kwargs)

    config = _config

    # Avoid double-counting when called inside _patched_generate's hook
    if getattr(self, _FORWARD_RUN_KEY, None) is not None:
        return _ORIGINAL_FORWARD(self, *args, **kwargs)

    if config.extract_hidden_states:
        kwargs.setdefault("output_hidden_states", True)
    if config.extract_attentions:
        kwargs.setdefault("output_attentions", True)
    kwargs.setdefault("return_dict", True)

    global _last_started_run_id
    run_id    = str(uuid.uuid4())
    input_ids = args[0] if args else kwargs.get("input_ids")
    input_len = input_ids.shape[-1] if input_ids is not None else 0
    _store.start_run(run_id, input_len)
    _last_started_run_id = run_id

    setattr(self, _FORWARD_RUN_KEY, run_id)
    try:
        output = _ORIGINAL_FORWARD(self, *args, **kwargs)
    finally:
        delattr(self, _FORWARD_RUN_KEY)

    # Treat the single output as step 0, then close the run
    item: Dict[str, Any] = {
        "kind": "step", "run_id": run_id, "step_idx": 0, "config": config,
    }
    if config.extract_hidden_states:
        hs = getattr(output, "hidden_states", None) or \
             getattr(output, "decoder_hidden_states", None)
        item["hidden_states"] = hs
        item["encoder_hidden_states"] = getattr(output, "encoder_hidden_states", None)
    if config.extract_attentions:
        att = getattr(output, "attentions", None) or \
              getattr(output, "decoder_attentions", None)
        item["attentions"] = att
        item["encoder_attentions"] = getattr(output, "encoder_attentions", None)
    if config.extract_logits:
        logits = getattr(output, "logits", None)
        item["logits"] = logits
        if logits is None:
            item["start_logits"] = getattr(output, "start_logits", None)
            item["end_logits"]   = getattr(output, "end_logits",   None)
        else:
            item["start_logits"] = item["end_logits"] = None
        out_type = type(output).__name__
        item["logit_mode"] = "last_token" if any(
            tag in out_type for tag in (
                "CausalLM", "Seq2SeqLM", "ConditionalGeneration",
                "SpeechSeq2Seq", "Vision2Seq",
            )
        ) else "full"

    _worker.enqueue_step(item)
    _worker.enqueue_end(_end_payload(config, run_id))
    return output


# ---------------------------------------------------------------------------
# run_forward — hook-based capture for a single forward pass
# ---------------------------------------------------------------------------

def run_diffusion(
    model: Any,
    steps: List[Dict[str, Any]],
) -> List[Optional[str]]:
    """
    Capture internals from a **diffusion model's iterative denoising loop**.

    Calls :func:`run_forward` once per denoising step and returns the
    corresponding run-ids in order.  Each step produces an independent
    :class:`~lamina.core.run.InternalsRun` in the store.

    Parameters
    ----------
    model : any object with ``register_forward_hook`` and ``__call__``
    steps : list[dict]
        One ``kwargs`` dict per denoising step; forwarded verbatim to
        ``run_forward(model, **step_kwargs)``.

    Returns
    -------
    list[str | None]
        One run-id per step (``None`` if the extractor is not initialised).

    Example
    -------
    ::

        # Build the partially-masked inputs for each diffusion step
        step_inputs = [
            {"input_ids": masked_ids_t}
            for masked_ids_t in diffusion_schedule(input_ids, n_steps=10)
        ]
        run_ids = lamina.run_diffusion(model, step_inputs)
        # Wait for finalization, then inspect each step:
        runs = [lamina.get_run(rid) for rid in run_ids]
    """
    run_ids: List[Optional[str]] = []
    for step_kwargs in steps:
        run_forward(model, **step_kwargs)
        run_ids.append(_last_started_run_id)
    return run_ids


def run_forward(model: Any, *args: Any, **kwargs: Any) -> Any:
    """
    Capture internals from a **single forward pass** using
    ``register_forward_hook`` — works correctly even when the model defines
    its own ``forward()`` override (which bypasses ``_patched_forward``).

    Use this for encoder-only models, classifiers, QA heads, masked LM, etc.

    Parameters
    ----------
    model : any object with ``register_forward_hook`` and ``__call__``
    *args, **kwargs
        Forwarded to ``model()``.  ``output_hidden_states``,
        ``output_attentions``, and ``return_dict`` are injected automatically.

    Returns
    -------
    ModelOutput
    """
    global _last_started_run_id

    if _config is None or _store is None or _worker is None:
        return model(*args, **kwargs)

    config = _config

    if config.extract_hidden_states:
        kwargs.setdefault("output_hidden_states", True)
    if config.extract_attentions:
        kwargs.setdefault("output_attentions", True)
    kwargs.setdefault("return_dict", True)

    input_ids = args[0] if args else kwargs.get("input_ids")
    input_len = input_ids.shape[-1] if input_ids is not None else 0

    global _next_thinking_end_token_id
    run_id = str(uuid.uuid4())
    thinking_token_id = _next_thinking_end_token_id
    _next_thinking_end_token_id = None
    _store.start_run(
        run_id,
        input_len,
        config=config,
        thinking_end_token_id=thinking_token_id,
    )
    _last_started_run_id = run_id

    step_counter = [0]
    hook_handle  = model.register_forward_hook(
        _make_forward_hook(run_id, config, _worker, step_counter)
    )

    try:
        output = model(*args, **kwargs)
    finally:
        hook_handle.remove()
        _worker.enqueue_end(_end_payload(config, run_id))

    return output


# ---------------------------------------------------------------------------
# Install / uninstall patches
# ---------------------------------------------------------------------------

def install_patches() -> None:
    """
    Monkey-patch ``transformers.GenerationMixin.generate`` and
    ``transformers.PreTrainedModel.forward``.
    Called once from ``lamina/__init__.py``.
    """
    global _ORIGINAL_GENERATE, _ORIGINAL_FORWARD

    try:
        import transformers
    except ImportError:
        warnings.warn(
            "[lamina] transformers is not installed — HF extractor inactive.",
            ImportWarning,
            stacklevel=3,
        )
        return

    GenerationMixin = transformers.generation.GenerationMixin
    PreTrainedModel = transformers.PreTrainedModel

    if GenerationMixin.generate is not _patched_generate:
        _ORIGINAL_GENERATE = GenerationMixin.generate
        GenerationMixin.generate = _patched_generate

    if PreTrainedModel.forward is not _patched_forward:
        _ORIGINAL_FORWARD = PreTrainedModel.forward
        PreTrainedModel.forward = _patched_forward


def uninstall_patches() -> None:
    """Restore original transformers methods (useful for testing)."""
    try:
        import transformers
    except ImportError:
        return

    if _ORIGINAL_GENERATE is not None:
        transformers.generation.GenerationMixin.generate = _ORIGINAL_GENERATE
    if _ORIGINAL_FORWARD is not None:
        transformers.PreTrainedModel.forward = _ORIGINAL_FORWARD
