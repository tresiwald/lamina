"""
lamina.extractors
=================
Framework-specific internals extractors.

Each extractor hooks into a particular inference framework (HuggingFace
Transformers, vLLM, …) and pushes captured tensors into the shared
core ring buffer via the BackgroundWorker queue.

Available extractors
--------------------
lamina.extractors.hf      — HuggingFace Transformers (GenerationMixin + hooks)
lamina.extractors.vllm    — vLLM (future)
"""
from .base import Extractor

__all__ = ["Extractor"]
