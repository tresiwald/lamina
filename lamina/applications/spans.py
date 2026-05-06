"""
Span types and offset-mapping utilities.

TextSpan
    Define a span by a *substring* of the input text.  The tokenizer resolves
    it to token indices at inference time using ``return_offsets_mapping=True``.

SpanSpec
    A span defined by explicit inclusive-exclusive **token indices**.

SpanResolutionError
    Raised when a TextSpan cannot be found or mapped.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Span types
# ---------------------------------------------------------------------------

@dataclass
class TextSpan:
    """
    Define a span by a **substring** of the input text.

    The tokenizer resolves this to token indices at inference time using
    ``return_offsets_mapping=True``, so the definition is stable across
    different tokenization settings.

    Parameters
    ----------
    text : str
        The exact substring to locate.
    label : str, optional
        Name for this span.  Used as the dict key when a list of TextSpan
        objects is passed to ``InternalsInstance.spans``.
    occurrence : int
        Which occurrence of the substring to use, 0-indexed (default: 0).
    strip : bool
        Strip leading/trailing whitespace before matching (default: True).
    """
    text: str
    label: Optional[str] = None
    occurrence: int = 0
    strip: bool = True

    def __post_init__(self):
        if self.occurrence < 0:
            raise ValueError(
                f"TextSpan occurrence must be >= 0 (got {self.occurrence})."
            )


class SpanResolutionError(ValueError):
    """Raised when a TextSpan substring cannot be found or mapped."""


@dataclass
class SpanSpec:
    """
    A span defined by explicit, inclusive-exclusive **token indices**.

    Parameters
    ----------
    start : int
        First token index, inclusive.  Negative values resolve relative to
        ``input_len`` (Python-style).
    end : int
        Last token index, exclusive.
    """
    start: int
    end: int

    def __post_init__(self):
        if self.start >= 0 and self.end >= 0 and self.end < self.start:
            raise ValueError(
                f"SpanSpec end ({self.end}) must be >= start ({self.start})"
            )

    @property
    def length(self) -> int:
        return self.end - self.start

    def as_slice(self) -> slice:
        return slice(self.start, self.end)


# Short-hand type alias
_SpanValue = Union[SpanSpec, TextSpan, Tuple[int, int], str]


def _normalise_span(value: _SpanValue) -> Union[SpanSpec, TextSpan]:
    """Convert raw user-supplied span value to SpanSpec or TextSpan."""
    if isinstance(value, (SpanSpec, TextSpan)):
        return value
    if isinstance(value, tuple):
        return SpanSpec(*value)
    if isinstance(value, str):
        return TextSpan(value)
    raise TypeError(
        f"Span value must be str, tuple, SpanSpec, or TextSpan; got {type(value)}"
    )


# ---------------------------------------------------------------------------
# Offset-mapping resolver
# ---------------------------------------------------------------------------

def _resolve_text_spans(
    text: str,
    spans: Dict[str, Union[SpanSpec, TextSpan]],
    tokenizer: Any,
) -> Dict[str, SpanSpec]:
    """
    Convert any TextSpan entries in *spans* to SpanSpec using the tokenizer's
    offset mapping.  SpanSpec entries are returned unchanged.
    """
    if not any(isinstance(v, TextSpan) for v in spans.values()):
        return spans  # type: ignore[return-value]

    try:
        enc = tokenizer(text, return_offsets_mapping=True)
    except (NotImplementedError, Exception) as exc:
        raise SpanResolutionError(
            "TextSpan resolution requires a fast tokenizer that supports "
            f"return_offsets_mapping=True.  Error: {exc}"
        ) from exc

    offsets: List[Tuple[int, int]] = enc["offset_mapping"]

    resolved: Dict[str, SpanSpec] = {}
    for name, span in spans.items():
        if isinstance(span, SpanSpec):
            resolved[name] = span
            continue

        search_text = span.text.strip() if span.strip else span.text
        occurrence  = span.occurrence

        char_start = -1
        search_from = 0
        for _ in range(occurrence + 1):
            char_start = text.find(search_text, search_from)
            if char_start == -1:
                break
            search_from = char_start + 1

        if char_start == -1:
            raise SpanResolutionError(
                f"TextSpan {search_text!r} occurrence={occurrence} not found "
                f"in text {text!r}"
            )
        char_end = char_start + len(search_text)

        tok_start: Optional[int] = None
        tok_end:   Optional[int] = None
        for i, (cs, ce) in enumerate(offsets):
            if cs == 0 and ce == 0:
                continue
            if tok_start is None and ce > char_start:
                tok_start = i
            if cs < char_end:
                tok_end = i + 1

        if tok_start is None or tok_end is None:
            raise SpanResolutionError(
                f"TextSpan {search_text!r} could not be mapped to any token. "
                f"char=[{char_start}:{char_end}], offsets={offsets}"
            )

        resolved[name] = SpanSpec(tok_start, tok_end)

    return resolved


# ---------------------------------------------------------------------------
# Span averaging
# ---------------------------------------------------------------------------

def _compute_span_means(
    run: Any,
    spans: Optional[Dict[str, SpanSpec]],
) -> Optional[Dict[str, np.ndarray]]:
    """
    For each SpanSpec, slice ``run.input_hidden_states`` and average over
    the span's token positions per layer.

    Returns None if no spans are defined or hidden states are unavailable.
    Returns dict: span_name → ``(num_layers, hidden_dim)`` float32 array.
    """
    if not spans or run.input_hidden_states is None:
        return None

    result: Dict[str, np.ndarray] = {}
    hidden_dim = run.input_hidden_states[0].shape[-1]
    input_len  = run.input_len

    for name, span in spans.items():
        start = span.start if span.start >= 0 else input_len + span.start
        end   = span.end   if span.end   >= 0 else input_len + span.end
        start = max(0, min(start, input_len))
        end   = max(start, min(end, input_len))

        layer_means: List[np.ndarray] = []
        for hs in run.input_hidden_states:
            sliced = hs[0, start:end, :]
            if sliced.shape[0] == 0:
                layer_means.append(np.zeros(hidden_dim, dtype=np.float32))
            else:
                layer_means.append(sliced.mean(axis=0).astype(np.float32))
        result[name] = np.stack(layer_means, axis=0)

    return result
