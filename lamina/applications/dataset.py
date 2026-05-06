"""
Dataset processing layer for lamina.

InternalsInstance
    One item: text/token-ids + arbitrary task properties + optional spans.

InternalsRecord
    Result for one instance: the full InternalsRun + task properties +
    per-span averaged hidden states.

InternalsDataset
    An ordered collection of instances with a ``.run()`` method that drives
    the model and collects InternalsRecord objects.

Model-type detection
--------------------
``run()`` auto-detects the model type and chooses the correct inference call:

* **Decoder-only / encoder-decoder** → ``model.generate()``
* **Encoder-only / classifiers / QA / MLM** → ``lamina.run_forward(model, …)``

Detection uses ``model.can_generate()`` (transformers ≥ 4.18) with a
class-name / config-level heuristic fallback.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .spans import (
    SpanSpec,
    TextSpan,
    SpanResolutionError,
    _SpanValue,
    _normalise_span,
    _resolve_text_spans,
    _compute_span_means,
)
from lamina.extractors.hf.model_detect import _model_can_generate


# ---------------------------------------------------------------------------
# InternalsInstance
# ---------------------------------------------------------------------------

@dataclass
class InternalsInstance:
    """
    One item to be processed by the model.

    Parameters
    ----------
    text : str or list[int]
        Raw text (tokenizer handles encoding) or pre-tokenised token IDs.
    properties : dict, optional
        Arbitrary task-specific metadata preserved verbatim in the output.
    spans : dict | list[TextSpan], optional
        Named regions of the input token sequence for per-layer averaging.

        Two forms are accepted:

        **Dict form** — keys are span names, values are any of: str,
        TextSpan, SpanSpec, or tuple[int, int].

        **List form** — a list of TextSpan objects each carrying a ``label``.
    """
    text: Union[str, List[int]]
    properties: Dict[str, Any] = field(default_factory=dict)
    spans: Optional[Union[Dict[str, _SpanValue], List[TextSpan]]] = None

    def __post_init__(self):
        if self.spans is None:
            return
        if isinstance(self.spans, list):
            spans_dict: Dict[str, _SpanValue] = {}
            for ts in self.spans:
                if not isinstance(ts, TextSpan):
                    raise TypeError(
                        f"List form of spans only accepts TextSpan objects; "
                        f"got {type(ts)}"
                    )
                if not ts.label:
                    raise ValueError(
                        f"TextSpan {ts.text!r} has no label.  "
                        "Set label= when passing spans as a list."
                    )
                if ts.label in spans_dict:
                    raise ValueError(f"Duplicate TextSpan label {ts.label!r}.")
                spans_dict[ts.label] = ts
            self.spans = spans_dict
        self.spans = {
            name: _normalise_span(v) for name, v in self.spans.items()
        }


# ---------------------------------------------------------------------------
# InternalsRecord
# ---------------------------------------------------------------------------

@dataclass
class InternalsRecord:
    """
    Extraction result for a single :class:`InternalsInstance`.

    Attributes
    ----------
    instance : InternalsInstance
    run : InternalsRun
    resolved_spans : dict[str, SpanSpec] | None
    span_hidden_states_mean : dict[str, np.ndarray] | None
        Per-span, per-layer mean hidden state.  Shape: ``(num_layers, hidden)``.
    """
    instance: InternalsInstance
    run: Any                   # InternalsRun (avoid circular import)
    resolved_spans: Optional[Dict[str, SpanSpec]] = None
    span_hidden_states_mean: Optional[Dict[str, np.ndarray]] = None

    @property
    def properties(self) -> Dict[str, Any]:
        return self.instance.properties

    @property
    def spans(self) -> Optional[Dict[str, Union[SpanSpec, TextSpan]]]:
        return self.instance.spans

    def __repr__(self) -> str:
        span_names = list(self.resolved_spans or self.instance.spans or {})
        return (
            f"InternalsRecord("
            f"properties={self.properties}, "
            f"spans={span_names}, "
            f"run={self.run})"
        )


# ---------------------------------------------------------------------------
# InternalsDataset
# ---------------------------------------------------------------------------

class InternalsDataset:
    """
    An ordered collection of :class:`InternalsInstance` objects.

    Constructors
    ------------
    Pass either a plain list of :class:`InternalsInstance` objects **or** a
    HuggingFace ``datasets.Dataset`` directly::

        # From a list
        dataset = InternalsDataset([
            InternalsInstance(text="Hello", properties={"label": 1}),
        ])

        # From an HF dataset
        from datasets import load_dataset
        ds = load_dataset("sst2", split="validation[:100]")
        dataset = InternalsDataset(ds, text_col="sentence", property_cols=["label"])
    """

    def __init__(
        self,
        source: Union[List[InternalsInstance], Any],
        *,
        text_col: str = "text",
        property_cols: Optional[List[str]] = None,
        spans: Optional[Dict[str, _SpanValue]] = None,
        spans_col: Optional[str] = None,
    ) -> None:
        if isinstance(source, list):
            self.instances: List[InternalsInstance] = source
        else:
            self.instances = self._instances_from_dataset(
                source, text_col, property_cols, spans, spans_col
            )

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> InternalsInstance:
        return self.instances[idx]

    @staticmethod
    def _instances_from_dataset(
        dataset: Any,
        text_col: str,
        property_cols: Optional[List[str]],
        spans: Optional[Dict[str, _SpanValue]],
        spans_col: Optional[str],
    ) -> List[InternalsInstance]:
        instances: List[InternalsInstance] = []
        all_cols = (
            set(dataset.column_names)
            if hasattr(dataset, "column_names")
            else set(next(iter(dataset)).keys())
        )
        exclude  = {text_col}
        if spans_col:
            exclude.add(spans_col)
        prop_cols = (
            property_cols if property_cols is not None
            else sorted(all_cols - exclude)
        )

        for row in dataset:
            text  = row[text_col]
            props = {col: row[col] for col in prop_cols if col in row}

            merged_spans: Optional[Dict[str, _SpanValue]] = dict(spans or {})
            if spans_col and spans_col in row and row[spans_col]:
                for span_name, span_val in row[spans_col].items():
                    if isinstance(span_val, str):
                        merged_spans[span_name] = TextSpan(span_val)
                    elif isinstance(span_val, (list, tuple)) and len(span_val) == 2:
                        merged_spans[span_name] = SpanSpec(
                            int(span_val[0]), int(span_val[1])
                        )
                    elif isinstance(span_val, dict):
                        merged_spans[span_name] = SpanSpec(
                            int(span_val["start"]), int(span_val["end"])
                        )
                    else:
                        merged_spans[span_name] = span_val

            instances.append(InternalsInstance(
                text=text,
                properties=props,
                spans=merged_spans if merged_spans else None,
            ))

        return instances

    @classmethod
    def from_hf_dataset(
        cls,
        dataset: Any,
        text_col: str = "text",
        property_cols: Optional[List[str]] = None,
        spans: Optional[Dict[str, _SpanValue]] = None,
        spans_col: Optional[str] = None,
    ) -> "InternalsDataset":
        """Backward-compatible classmethod constructor."""
        return cls(
            dataset,
            text_col=text_col,
            property_cols=property_cols,
            spans=spans,
            spans_col=spans_col,
        )

    def run(
        self,
        model: Any,
        tokenizer: Any,
        generate_kwargs: Optional[Dict[str, Any]] = None,
        finalize_timeout: float = 60.0,
        verbose: bool = True,
    ) -> List[InternalsRecord]:
        """
        Process every instance and return a list of :class:`InternalsRecord`.

        The inference call is chosen automatically:

        * **Decoder-only / encoder-decoder** → ``model.generate()``
        * **Encoder-only / classifiers / QA / MLM** → ``run_forward(model, …)``

        Parameters
        ----------
        model : PreTrainedModel
        tokenizer : PreTrainedTokenizer / PreTrainedTokenizerFast
        generate_kwargs : dict, optional
            Forwarded to ``model.generate()``.  Defaults to
            ``{"max_new_tokens": 1}``.  Ignored for non-generative models.
        finalize_timeout : float
            Seconds to wait for the background worker per run.
        verbose : bool
        """
        import lamina
        from lamina.extractors.hf.extractor import (
            run_forward,
            _last_started_run_id as _get_last_run_id,
        )

        use_generate = _model_can_generate(model)
        gkw: Dict[str, Any] = dict(generate_kwargs or {})
        if use_generate:
            gkw.setdefault("max_new_tokens", 1)

        records: List[InternalsRecord] = []
        n = len(self.instances)

        for idx, instance in enumerate(self.instances):
            if verbose:
                _prop_str = ", ".join(
                    f"{k}={v!r}"
                    for k, v in list(instance.properties.items())[:3]
                )
                print(f"  [{idx + 1:>{len(str(n))}}/{n}]  {_prop_str}")

            # ── Encode ────────────────────────────────────────────────────────
            if isinstance(instance.text, str):
                enc = tokenizer(instance.text, return_tensors="pt")
                enc = {k: v.to(model.device) for k, v in enc.items()}
                input_ids = enc["input_ids"]
            else:
                import torch
                input_ids = torch.tensor([instance.text], device=model.device)
                enc = {"input_ids": input_ids}

            # ── Resolve text spans ────────────────────────────────────────────
            resolved_spans: Optional[Dict[str, SpanSpec]] = None
            if instance.spans and isinstance(instance.text, str):
                resolved_spans = _resolve_text_spans(
                    instance.text, instance.spans, tokenizer
                )
            elif instance.spans:
                bad = [k for k, v in instance.spans.items()
                       if isinstance(v, TextSpan)]
                if bad:
                    raise SpanResolutionError(
                        f"TextSpan cannot be used with pre-tokenised input "
                        f"(spans: {bad}). Use SpanSpec with explicit indices."
                    )
                resolved_spans = instance.spans  # type: ignore[assignment]

            # ── Inference ─────────────────────────────────────────────────────
            if use_generate:
                kwargs = dict(gkw)
                if "attention_mask" in enc:
                    kwargs.setdefault("attention_mask", enc["attention_mask"])
                model.generate(input_ids, **kwargs)
            else:
                run_forward(model, **enc)

            # ── Wait for background worker ────────────────────────────────────
            from lamina.extractors.hf import extractor as _ext
            last_run_id = _ext._last_started_run_id
            if last_run_id is None:
                raise RuntimeError(
                    "lamina: no run was recorded — is the plugin active?"
                )

            run = lamina.wait_for_run(last_run_id, timeout=finalize_timeout)
            if run is None:
                raise TimeoutError(
                    f"lamina: run {last_run_id!r} did not finalise "
                    f"within {finalize_timeout}s"
                )

            # ── Span averages ─────────────────────────────────────────────────
            span_means = _compute_span_means(run, resolved_spans)
            records.append(InternalsRecord(
                instance=instance,
                run=run,
                resolved_spans=resolved_spans,
                span_hidden_states_mean=span_means,
            ))

        if verbose:
            print(f"  Extraction complete — {len(records)} records.")

        return records
