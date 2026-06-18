"""
Subword-to-word alignment utilities.

Converts between token-level positions (as captured by lamina) and word-level
positions (as produced by NLP labeling pipelines such as spaCy).

Quick-start
-----------
::

    from lamina.applications.alignment import (
        word_ids,
        word_to_token_spans,
        align_word_labels,
    )

    text = "The cat sat"
    wids = word_ids(text, tokenizer)
    # e.g. [-1, 0, 1, 2, -1]  (-1 = special token like BOS/EOS)

    # After running spaCy: doc = nlp(text)
    pos_by_word = {i: tok.pos_ for i, tok in enumerate(doc)}
    pos_by_token = align_word_labels(text, tokenizer, pos_by_word)
    # e.g. [None, 'DET', 'NOUN', 'VERB', None]

Word-index convention
---------------------
* Words are segmented by whitespace (spaCy / HuggingFace word boundary).
* Indices are 0-based and match ``spaCy Doc`` token indices when spaCy is
  used with ``tokenizer.is_space`` semantics.
* Special tokens (``[CLS]``, ``[SEP]``, ``<s>``, ``</s>``, …) are assigned
  word index ``-1``.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# word_ids
# ---------------------------------------------------------------------------

def word_ids(text: str, tokenizer: Any) -> np.ndarray:
    """
    Return the word index for every token in *text*, shape ``(num_tokens,)``.

    Special tokens (``[CLS]``, ``<s>``, …) map to ``-1``.
    All sub-tokens of the same word share the same word index.

    For HuggingFace **fast** tokenizers the result is computed from
    ``BatchEncoding.word_ids()`` (always available and always correct).
    For **slow** tokenizers a fallback based on ``offset_mapping`` and
    whitespace word segmentation is used.

    Parameters
    ----------
    text : str
    tokenizer : PreTrainedTokenizer | PreTrainedTokenizerFast

    Returns
    -------
    np.ndarray of dtype int32, shape (num_tokens,)
        ``wids[i] == -1``  →  token ``i`` is a special token.
        ``wids[i] == k``   →  token ``i`` belongs to word ``k``.
    """
    enc = tokenizer(text, return_offsets_mapping=True)

    # Fast tokenizer path — always prefer it
    try:
        raw = enc.word_ids()
        return np.array([-1 if w is None else w for w in raw], dtype=np.int32)
    except AttributeError:
        pass  # slow tokenizer — fall through

    return _word_ids_from_offsets(text, enc["offset_mapping"])


def _word_ids_from_offsets(
    text: str,
    offsets: List[Tuple[int, int]],
) -> np.ndarray:
    """Fallback for slow tokenizers: derive word indices from char offsets."""
    # Build a char → word_idx map based on whitespace boundaries
    char_to_word: Dict[int, int] = {}
    word_idx = -1
    in_word = False
    for i, ch in enumerate(text):
        if ch.isspace():
            in_word = False
        else:
            if not in_word:
                word_idx += 1
                in_word = True
            char_to_word[i] = word_idx

    result: List[int] = []
    for char_start, char_end in offsets:
        if char_start == 0 and char_end == 0:
            result.append(-1)  # special token (no span in source text)
        else:
            result.append(char_to_word.get(char_start, -1))
    return np.array(result, dtype=np.int32)


# ---------------------------------------------------------------------------
# word_to_token_spans
# ---------------------------------------------------------------------------

def word_to_token_spans(
    text: str,
    tokenizer: Any,
) -> Dict[int, Tuple[int, int]]:
    """
    Return a mapping ``{word_idx: (token_start, token_end)}`` for every word.

    The span is inclusive-exclusive in token space so it can be passed
    directly to :class:`~lamina.applications.spans.SpanSpec`.

    Parameters
    ----------
    text : str
    tokenizer : PreTrainedTokenizer | PreTrainedTokenizerFast

    Returns
    -------
    dict[int, tuple[int, int]]
        ``spans[k] == (start, end)``  →  word ``k`` occupies tokens
        ``[start, end)`` (standard Python slice notation).

    Example
    -------
    ::

        spans = word_to_token_spans("The cat sat", tokenizer)
        # {0: (1, 2), 1: (2, 3), 2: (3, 4)}  — skipping BOS (idx 0)
    """
    wids = word_ids(text, tokenizer)
    spans: Dict[int, Tuple[int, int]] = {}
    for tok_idx, w in enumerate(wids.tolist()):
        if w < 0:
            continue
        if w not in spans:
            spans[w] = (tok_idx, tok_idx + 1)
        else:
            spans[w] = (spans[w][0], tok_idx + 1)
    return spans


# ---------------------------------------------------------------------------
# align_word_labels
# ---------------------------------------------------------------------------

def align_word_labels(
    text: str,
    tokenizer: Any,
    word_labels: Dict[int, Any],
    default: Any = None,
) -> List[Any]:
    """
    Map word-level labels onto token positions.

    Parameters
    ----------
    text : str
    tokenizer : PreTrainedTokenizer | PreTrainedTokenizerFast
    word_labels : dict[int, Any]
        ``{word_idx: label}`` — typically produced by spaCy or any other
        word-level NLP pipeline.
    default : Any
        Label assigned to special tokens and words not present in
        *word_labels*.  Defaults to ``None``.

    Returns
    -------
    list[Any], length == number of tokens
        ``result[i] == word_labels[w]`` where ``w`` is the word index for
        token ``i``, or *default* for special tokens (word index ``-1``).

    Example
    -------
    ::

        import spacy
        from lamina.applications.alignment import align_word_labels

        nlp = spacy.load("en_core_web_sm")
        text = "The cat sat"
        doc  = nlp(text)

        pos_by_word = {i: tok.pos_ for i, tok in enumerate(doc)}
        pos_by_tok  = align_word_labels(text, tokenizer, pos_by_word)
        # [None, 'DET', 'NOUN', 'VERB', None]  (None = BOS / EOS)
    """
    wids = word_ids(text, tokenizer)
    return [
        word_labels.get(int(w), default) if w >= 0 else default
        for w in wids.tolist()
    ]


# ---------------------------------------------------------------------------
# Convenience: build label array for probing
# ---------------------------------------------------------------------------

def token_label_array(
    records: List[Any],
    text_fn: Callable[[Any], str],
    tokenizer: Any,
    word_labels_fn: Callable[[Any], Dict[int, Any]],
    label_to_int: Optional[Dict[str, int]] = None,
) -> Tuple[np.ndarray, Dict[int, str]]:
    """
    Build a ``(N, max_seq_len)`` integer label matrix from a list of records.

    Useful for building token-level classification datasets from lamina
    records where each token needs its own label (POS, dep, NER, …).

    Parameters
    ----------
    records : list[InternalsRecord]
    text_fn : callable(record) → str
        Extract the text from a record.
    tokenizer : tokenizer
    word_labels_fn : callable(record) → dict[int, Any]
        Produce ``{word_idx: label}`` for a record.  Typically built from a
        spaCy doc over ``record.generated_text`` or ``record.instance.text``.
    label_to_int : dict[str, int] | None
        If not supplied, all unique string labels are sorted and assigned
        integer indices automatically.

    Returns
    -------
    labels : np.ndarray of shape (N, max_seq_len), dtype int32
        Rows are records; columns are token positions.  Positions without a
        valid label are filled with ``-1``.
    id_to_label : dict[int, str]
        Reverse mapping from integer index to original label string.
    """
    all_labels_per_record: List[List[Any]] = []
    for rec in records:
        text = text_fn(rec)
        wl   = word_labels_fn(rec)
        toks = align_word_labels(text, tokenizer, wl, default=None)
        all_labels_per_record.append(toks)

    # Build label vocabulary if not supplied
    if label_to_int is None:
        vocab: List[str] = sorted({
            lab for row in all_labels_per_record for lab in row if lab is not None
        })
        label_to_int = {lab: i for i, lab in enumerate(vocab)}
    id_to_label = {v: k for k, v in label_to_int.items()}

    max_len = max(len(row) for row in all_labels_per_record)
    out = np.full((len(records), max_len), fill_value=-1, dtype=np.int32)
    for i, row in enumerate(all_labels_per_record):
        for j, lab in enumerate(row):
            if lab is not None and lab in label_to_int:
                out[i, j] = label_to_int[lab]

    return out, id_to_label
