"""
Abstract base class for framework-specific internals extractors.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Extractor(ABC):
    """
    Protocol for framework-specific internals extractors.

    An extractor is responsible for:

    1. **Patching** the inference framework at import time so that every
       inference call automatically captures internals (``patch()``).
    2. **Providing a direct capture path** (``run_forward()``) for models
       that do not use the framework's generation loop (encoder-only,
       classifiers, etc.).
    3. **Unpatching** for test isolation (``unpatch()``).

    The extractor does *not* own the ring buffer or the worker thread —
    those live in ``lamina.core`` and are passed in at construction time.

    Concrete implementations
    ------------------------
    ``lamina.extractors.hf.HFExtractor``
        Patches ``transformers.GenerationMixin.generate`` and installs
        ``register_forward_hook`` on each model instance.

    Future
    ------
    ``lamina.extractors.vllm.VLLMExtractor``   (not yet implemented)
    """

    @abstractmethod
    def patch(self) -> None:
        """
        Install framework-level monkey-patches.
        Called once at import time by ``lamina/__init__.py``.
        """

    @abstractmethod
    def unpatch(self) -> None:
        """
        Restore original framework methods.
        Primarily used in tests to reset state between runs.
        """

    @abstractmethod
    def run_forward(self, model: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Capture internals from a **single forward pass**.

        Use this for non-generative models — encoder-only (BERT, RoBERTa),
        sequence/token classifiers, QA heads, masked-LM, etc. — that do
        not have a generation loop.

        Parameters
        ----------
        model : any object with ``register_forward_hook`` and ``__call__``
        *args, **kwargs
            Forwarded verbatim to ``model()``.  Framework-specific kwargs
            such as ``output_hidden_states`` and ``return_dict`` are
            injected automatically.

        Returns
        -------
        ModelOutput (or whatever the model returns)
        """
