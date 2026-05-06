"""
lamina.extractors.hf
====================
HuggingFace Transformers extractor.

Patches ``transformers.GenerationMixin.generate`` once at import time
and uses ``register_forward_hook`` on model instances for fine-grained
per-step capture.

Public surface
--------------
``run_forward(model, *args, **kwargs)``
    Capture internals from a single forward pass (non-generative models).

``register_model(model)``
    Copy LM-head weights to CPU for logit-lens computation.

``install_patches()`` / ``uninstall_patches()``
    Used internally and in tests.
"""
from .extractor import (
    run_forward,
    install_patches,
    uninstall_patches,
    _initialise,
    _last_started_run_id,
    _ORIGINAL_GENERATE,
)
from .lm_head import register_model
from .model_detect import _model_can_generate

__all__ = [
    "run_forward",
    "install_patches",
    "uninstall_patches",
    "register_model",
    "_model_can_generate",
    "_initialise",
]
