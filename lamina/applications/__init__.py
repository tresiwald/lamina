"""
lamina.applications
===================
Higher-level abstractions built on top of ``lamina.core`` and
``lamina.extractors``.

``lamina.applications.spans``
    Span types (TextSpan, SpanSpec) and offset-mapping resolver used to
    average hidden states over named regions of the input.

``lamina.applications.dataset``
    InternalsDataset / InternalsInstance / InternalsRecord — the primary
    batch-processing API for running a model over a collection of texts
    and collecting structured extraction results.
"""
from .spans import SpanSpec, TextSpan, SpanResolutionError
from .dataset import InternalsInstance, InternalsRecord, InternalsDataset

__all__ = [
    "SpanSpec",
    "TextSpan",
    "SpanResolutionError",
    "InternalsInstance",
    "InternalsRecord",
    "InternalsDataset",
]
