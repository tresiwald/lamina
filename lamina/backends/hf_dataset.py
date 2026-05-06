"""
HuggingFace datasets backend.

Converts ``InternalsRecord`` lists to a HuggingFace ``datasets.Dataset``
for Arrow serialisation, easy ``push_to_hub()``, and seamless integration
with the HF ecosystem.

Requires::

    pip install lamina[backends]   # pulls in `datasets`
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .base import Backend
from .filesystem import _build_arrays


class HFDatasetBackend(Backend):
    """
    Converts records to a HuggingFace ``datasets.Dataset``.

    Parameters
    ----------
    save_full_hidden_states : bool
        Include full 3-D ``input_hidden_states`` / ``output_hidden_states``
        arrays.  Off by default because they can be large.
    """

    def __init__(self, save_full_hidden_states: bool = False) -> None:
        self._save_full = save_full_hidden_states

    def write(self, records: List[Any], **kwargs: Any) -> Any:
        """Return a ``datasets.Dataset`` (does not write to disk)."""
        return to_hf_dataset(
            records,
            save_full_hidden_states=kwargs.get(
                "save_full_hidden_states", self._save_full
            ),
        )

    @classmethod
    def read(cls, source: Any, **kwargs: Any) -> Tuple[List[Dict], List[Dict]]:  # type: ignore[override]
        """
        Load a previously saved HF dataset from disk.

        Parameters
        ----------
        source : str
            Path passed to ``datasets.load_from_disk()``.
        """
        try:
            from datasets import load_from_disk
        except ImportError as exc:
            raise ImportError(
                "HFDatasetBackend.read() requires the 'datasets' package. "
                "Install with: pip install lamina[backends]"
            ) from exc

        ds = load_from_disk(source)
        arrays_list = []
        meta_list   = []
        for row in ds:
            arrays = {k: v for k, v in row.items()
                      if hasattr(v, "__len__") and not isinstance(v, str)}
            meta   = {k: v for k, v in row.items() if k not in arrays}
            arrays_list.append(arrays)
            meta_list.append(meta)
        return arrays_list, meta_list


# ---------------------------------------------------------------------------
# Module-level convenience function (backward-compatible)
# ---------------------------------------------------------------------------

def to_hf_dataset(
    records: List[Any],
    save_full_hidden_states: bool = False,
) -> Any:
    """
    Convert a list of ``InternalsRecord`` to a HuggingFace ``datasets.Dataset``.

    Requires ``pip install lamina[backends]``.

    Schema
    ------
    Each row contains all fields from ``record.properties`` at the top level,
    plus: ``run_id``, ``input_len``, ``num_layers``, ``num_output_tokens``,
    ``input_hidden_states_mean``, ``output_hidden_states_mean``, ``logits``,
    ``span_{name}`` (per span), ``span_{name}_start`` / ``_end`` (token indices).

    Returns
    -------
    datasets.Dataset
    """
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError(
            "to_hf_dataset() requires the 'datasets' package. "
            "Install with: pip install lamina[backends]"
        ) from exc

    rows: List[Dict[str, Any]] = []
    for record in records:
        arrays, _ = _build_arrays(
            record,
            save_attentions=False,
            save_full_hidden_states=save_full_hidden_states,
        )
        run = record.run
        row: Dict[str, Any] = {
            "run_id":            run.run_id,
            "input_len":         run.input_len,
            "num_layers":        run.num_layers,
            "num_output_tokens": run.num_output_tokens,
        }
        for k, v in record.properties.items():
            row[k] = v
        for key, arr in arrays.items():
            row[key] = arr.tolist()
        for name, span in (record.resolved_spans or {}).items():
            row[f"span_{name}_start"] = span.start
            row[f"span_{name}_end"]   = span.end
        rows.append(row)

    return Dataset.from_list(rows)
