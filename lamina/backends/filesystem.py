"""
Filesystem backend — serialise InternalsRecord lists to disk.

Layout
------
``write(records, outdir)`` produces::

    outdir/
        metadata.jsonl      one JSON object per line, one per record
        00000.npz           NumPy archive for record 0
        00001.npz           NumPy archive for record 1
        …

metadata.jsonl fields
---------------------
Each line is a JSON object with:

    index               int
    run_id              str
    input_len           int
    num_layers          int
    num_output_tokens   int
    properties          dict
    spans               dict  {name: {start, end}}
    arrays              dict  {array_key: shape}

.npz arrays (batch dimension squeezed)
---------------------------------------
    input_hidden_states_mean    float32  (num_layers, hidden)
    output_hidden_states_mean   float32  (num_layers, hidden)
    logits                      float32  (steps, vocab)
    logit_lens                  float32  (num_layers, input_len, vocab)
    attentions_step{N}          float32  (num_layers, seq_q, seq_k)
    span_{name}                 float32  (num_layers, hidden)
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import Backend


class FilesystemBackend(Backend):
    """
    Writes records to a directory of ``.npz`` files + ``metadata.jsonl``.

    Parameters
    ----------
    outdir : str
        Directory to write into.  Created if it does not exist.
    save_attentions : bool
        When True, include per-step attention arrays.  Can be large.
    save_full_hidden_states : bool
        When True, include full 3-D hidden state arrays in addition to means.
    """

    def __init__(
        self,
        outdir: str,
        save_attentions: bool = False,
        save_full_hidden_states: bool = False,
    ) -> None:
        self._outdir = outdir
        self._save_attentions = save_attentions
        self._save_full_hidden_states = save_full_hidden_states

    def write(self, records: List[Any], **kwargs: Any) -> None:
        """Persist *records* to ``self._outdir``."""
        _write(
            records,
            self._outdir,
            save_attentions=kwargs.get("save_attentions", self._save_attentions),
            save_full_hidden_states=kwargs.get(
                "save_full_hidden_states", self._save_full_hidden_states
            ),
        )

    @classmethod
    def read(cls, source: str, **kwargs: Any) -> Tuple[List[Dict], List[Dict]]:
        """Load records from a previously written directory."""
        return _read(source)


# ---------------------------------------------------------------------------
# Module-level convenience functions (backward-compatible surface)
# ---------------------------------------------------------------------------

def dump(
    records: List[Any],
    outdir: str,
    save_attentions: bool = False,
    save_full_hidden_states: bool = False,
) -> None:
    """
    Persist a list of :class:`~lamina.applications.dataset.InternalsRecord`
    to disk.

    Equivalent to ``FilesystemBackend(outdir).write(records)``.
    """
    _write(records, outdir, save_attentions, save_full_hidden_states)


def load(outdir: str) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, Any]]]:
    """
    Load a previously dumped directory.

    Returns
    -------
    arrays_list : list[dict[str, np.ndarray]]
    metadata_list : list[dict]
    """
    return _read(outdir)


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------

def _write(
    records: List[Any],
    outdir: str,
    save_attentions: bool,
    save_full_hidden_states: bool,
) -> None:
    os.makedirs(outdir, exist_ok=True)
    meta_path = os.path.join(outdir, "metadata.jsonl")

    with open(meta_path, "w", encoding="utf-8") as meta_f:
        for idx, record in enumerate(records):
            arrays, array_shapes = _build_arrays(
                record, save_attentions, save_full_hidden_states
            )
            npz_path = os.path.join(outdir, f"{idx:05d}.npz")
            np.savez_compressed(npz_path, **arrays)
            meta_f.write(json.dumps(_build_meta(idx, record, array_shapes),
                                    ensure_ascii=False) + "\n")


def _read(outdir: str) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, Any]]]:
    meta_path = os.path.join(outdir, "metadata.jsonl")
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata_list = [json.loads(line) for line in f if line.strip()]

    arrays_list: List[Dict[str, np.ndarray]] = []
    for meta in metadata_list:
        npz_path = os.path.join(outdir, f"{meta['index']:05d}.npz")
        npz = np.load(npz_path)
        arrays_list.append({k: npz[k] for k in npz.files})

    return arrays_list, metadata_list


def _build_arrays(
    record: Any,
    save_attentions: bool,
    save_full_hidden_states: bool,
) -> Tuple[Dict[str, np.ndarray], Dict[str, tuple]]:
    run    = record.run
    arrays: Dict[str, np.ndarray] = {}

    # ── Hidden-state means ────────────────────────────────────────────────────
    if run.input_hidden_states_mean is not None:
        arrays["input_hidden_states_mean"] = run.input_hidden_states_mean[:, 0, :]
    if run.output_hidden_states_mean is not None:
        arrays["output_hidden_states_mean"] = run.output_hidden_states_mean[:, 0, :]

    # ── Full hidden states (optional) ─────────────────────────────────────────
    if save_full_hidden_states:
        if run.input_hidden_states is not None:
            arrays["input_hidden_states"] = np.stack(
                [hs[0] for hs in run.input_hidden_states], axis=0
            )
        if run.output_hidden_states is not None:
            valid = [hs for hs in run.output_hidden_states
                     if hs.ndim == 3 and hs.shape[1] > 0]
            if valid:
                arrays["output_hidden_states"] = np.stack(
                    [hs[0] for hs in run.output_hidden_states], axis=0
                )

    # ── Logits ───────────────────────────────────────────────────────────────
    if run.logits is not None:
        arrays["logits"] = run.logits[:, 0, :]

    # ── Logit lens ───────────────────────────────────────────────────────────
    if run.logit_lens is not None:
        arrays["logit_lens"] = np.stack(
            [ll[0] for ll in run.logit_lens], axis=0
        )

    # ── Attentions (optional) ─────────────────────────────────────────────────
    if save_attentions and run.attentions is not None:
        for step_idx, step_attn in enumerate(run.attentions):
            arrays[f"attentions_step{step_idx}"] = np.stack(
                [a[0] for a in step_attn], axis=0
            )

    # ── Span averages ─────────────────────────────────────────────────────────
    if record.span_hidden_states_mean is not None:
        for span_name, span_arr in record.span_hidden_states_mean.items():
            arrays[f"span_{span_name}"] = span_arr

    shape_dict = {k: tuple(v.shape) for k, v in arrays.items()}
    return arrays, shape_dict


def _build_meta(
    idx: int,
    record: Any,
    array_shapes: Dict[str, tuple],
) -> Dict[str, Any]:
    run = record.run
    resolved = record.resolved_spans or {}
    spans_meta = {
        name: {"start": s.start, "end": s.end}
        for name, s in resolved.items()
    }
    return {
        "index":             idx,
        "run_id":            run.run_id,
        "input_len":         run.input_len,
        "num_layers":        run.num_layers,
        "num_output_tokens": run.num_output_tokens,
        "properties":        record.properties,
        "spans":             spans_meta,
        "arrays":            {k: list(v) for k, v in array_shapes.items()},
    }
