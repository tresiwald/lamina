"""
Dataset processing layer for lamina.

InternalsInstance
    One item: text/token-ids + arbitrary task properties + optional spans
    + an optional per-instance system prompt + an optional thinking end token.

InternalsRecord
    Result for one instance: the full InternalsRun + task properties +
    per-span averaged hidden states + optional thinking-end hidden state.

InternalsDataset
    An ordered collection of instances with a ``.run()`` method that drives
    the model and collects InternalsRecord objects.

System prompts
--------------
System prompts are applied via ``tokenizer.apply_chat_template`` so that the
correct chat format (``<|system|>``, ``[INST]``, ``<|im_start|>``, …) is used
for the loaded model.  A plain-concatenation fallback is used for tokenizers
without a registered chat template.

Priority (highest first):

1. ``instance.system_prompt``  — per-instance, set on :class:`InternalsInstance`
2. ``run(system_prompt=…)``    — default for the whole :meth:`InternalsDataset.run` call
3. No system prompt            — plain ``tokenizer(text, …)``

TextSpan resolution works correctly even with system prompts because spans are
resolved against the *fully formatted* string that was actually tokenised, not
just the raw user text.

Thinking models
---------------
For models that produce a visible thinking phase (DeepSeek-R1, Qwen3, …) the
hidden state at the **last thinking token** can be extracted automatically.

Set ``thinking_end_token`` (e.g. ``"</think>"``) at instance or run level.
After generation lamina scans the output token IDs for the last occurrence of
that token and slices ``output_hidden_states`` at that position.  The result is
stored in ``InternalsRecord.thinking_hidden_state`` with shape
``(num_layers, hidden_dim)``.

Priority (highest first):

1. ``instance.thinking_end_token`` — per-instance override
2. ``run(thinking_end_token=…)``   — default for the whole run
3. No extraction                   — ``thinking_hidden_state`` is ``None``

Model-type detection
--------------------
``run()`` auto-detects the model type and chooses the correct inference call:

* **Decoder-only / encoder-decoder** → ``model.generate()``
* **Encoder-only / classifiers / QA / MLM** → ``lamina.run_forward(model, …)``
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

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

    system_prompt : str, optional
        System prompt prepended to *this* instance when building the model
        input.  Overrides the ``system_prompt`` argument of
        :meth:`InternalsDataset.run` for this specific instance.

        When set, the input is formatted via
        ``tokenizer.apply_chat_template([{"role": "system", …},
        {"role": "user", …}])`` so the correct template for the loaded
        model is used automatically.

        Example::

            InternalsInstance(
                text="Is Paris in France?",
                system_prompt="Answer in one word.",
                properties={"label": "yes"},
            )
    thinking_end_token : str, optional
        The token string that marks the end of the thinking phase for
        reasoning models (e.g. ``"</think>"`` for DeepSeek-R1 / Qwen3).

        After generation lamina scans the output token IDs for the *last*
        occurrence of this token and extracts
        ``output_hidden_states[:, pos, :]`` at that position.  The result
        is stored in :attr:`InternalsRecord.thinking_hidden_state` with
        shape ``(num_layers, hidden_dim)``.

        Overrides the ``thinking_end_token`` argument of
        :meth:`InternalsDataset.run` for this specific instance.

        Example::

            InternalsInstance(
                text="How many r's in 'strawberry'?",
                thinking_end_token="</think>",
                properties={"label": 3},
            )
    """
    text: Union[str, List[int]]
    properties: Dict[str, Any] = field(default_factory=dict)
    spans: Optional[Union[Dict[str, _SpanValue], List[TextSpan]]] = None
    system_prompt: Optional[str] = None
    thinking_end_token: Optional[str] = None

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
    thinking_hidden_state : np.ndarray | None
        Hidden state at the last thinking token (e.g. ``</think>``).
        Shape: ``(num_layers, hidden_dim)``.  ``None`` when no
        ``thinking_end_token`` was set or the token was not found in the
        generated output.
    thinking_end_token_pos : int | None
        Zero-based index into the *output* token sequence where the last
        thinking token was found.  ``None`` when ``thinking_hidden_state``
        is ``None``.
    """
    instance: InternalsInstance
    run: Any                   # InternalsRun (avoid circular import)
    resolved_spans: Optional[Dict[str, SpanSpec]] = None
    span_hidden_states_mean: Optional[Dict[str, np.ndarray]] = None
    thinking_hidden_state: Optional[np.ndarray] = None
    thinking_end_token_pos: Optional[int] = None

    @property
    def properties(self) -> Dict[str, Any]:
        return self.instance.properties

    @property
    def spans(self) -> Optional[Dict[str, Union[SpanSpec, TextSpan]]]:
        return self.instance.spans

    def __repr__(self) -> str:
        span_names = list(self.resolved_spans or self.instance.spans or {})
        think_str = (
            f", thinking_pos={self.thinking_end_token_pos}"
            if self.thinking_end_token_pos is not None else ""
        )
        return (
            f"InternalsRecord("
            f"properties={self.properties}, "
            f"spans={span_names}"
            f"{think_str}, "
            f"run={self.run})"
        )


# ---------------------------------------------------------------------------
# System-prompt formatting helper
# ---------------------------------------------------------------------------

def _apply_system_prompt(
    text: str,
    system_prompt: str,
    tokenizer: Any,
) -> str:
    """
    Format *text* with *system_prompt* using the tokenizer's chat template.

    Returns the fully formatted string (not yet tokenised) so it can be
    passed to ``tokenizer(…, return_tensors="pt")`` and to
    ``_resolve_text_spans`` for TextSpan resolution.

    Falls back to ``"{system_prompt}\\n\\n{text}"`` when the tokenizer has
    no ``chat_template`` attribute or ``apply_chat_template`` raises.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": text},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        warnings.warn(
            "[lamina] tokenizer.apply_chat_template() failed — "
            "falling back to plain concatenation for system prompt.",
            stacklevel=4,
        )
        return f"{system_prompt}\n\n{text}"


# ---------------------------------------------------------------------------
# Thinking-token extraction helper
# ---------------------------------------------------------------------------

def _extract_thinking_hidden_state(
    run: Any,
    generated_ids: "np.ndarray",   # shape (output_len,)  — 1-D, batch stripped
    thinking_end_token_id: int,
) -> "tuple[Optional[np.ndarray], Optional[int]]":
    """
    Find the last occurrence of *thinking_end_token_id* in *generated_ids*
    and return the hidden state at that position across all layers.

    Parameters
    ----------
    run : InternalsRun
        A finalised run with ``output_hidden_states`` populated.
    generated_ids : np.ndarray
        1-D array of generated token IDs (prompt tokens excluded).
    thinking_end_token_id : int
        Token ID to search for.

    Returns
    -------
    (hidden_state, pos) or (None, None)
        hidden_state : np.ndarray of shape ``(num_layers, hidden_dim)``
        pos          : int, 0-based index in the output sequence
    """
    if run.output_hidden_states is None:
        return None, None

    # Find the last occurrence of the token in the output sequence
    positions = np.where(generated_ids == thinking_end_token_id)[0]
    if positions.size == 0:
        return None, None

    pos = int(positions[-1])
    num_layers = len(run.output_hidden_states)
    output_len = run.output_hidden_states[0].shape[1]

    if pos >= output_len:
        return None, None

    layer_vecs: List[np.ndarray] = []
    for hs in run.output_hidden_states:        # hs: (batch, output_len, hidden)
        layer_vecs.append(hs[0, pos, :].astype(np.float32))

    return np.stack(layer_vecs, axis=0), pos   # (num_layers, hidden_dim)


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
        dataset = InternalsDataset(
            ds,
            text_col="sentence",
            property_cols=["label"],
            system_prompt_col="system",   # optional: per-row system prompts
        )

    System prompts
    --------------
    Three ways to attach system prompts (highest priority first):

    1. ``InternalsInstance(system_prompt="…")`` — per-instance
    2. ``dataset.run(…, system_prompt="…")`` — default for the whole run
    3. ``system_prompt_col`` in the HF dataset constructor — per-row from a
       column, stored as ``instance.system_prompt``

    Thinking models
    ---------------
    Pass ``thinking_end_token="</think>"`` to ``run()`` (or set it on
    individual instances) to extract the hidden state at the last thinking
    token.  See :meth:`run` for details.
    """

    def __init__(
        self,
        source: Union[List[InternalsInstance], Any],
        *,
        text_col: str = "text",
        property_cols: Optional[List[str]] = None,
        spans: Optional[Dict[str, _SpanValue]] = None,
        spans_col: Optional[str] = None,
        system_prompt_col: Optional[str] = None,
    ) -> None:
        if isinstance(source, list):
            self.instances: List[InternalsInstance] = source
        else:
            self.instances = self._instances_from_dataset(
                source, text_col, property_cols, spans, spans_col,
                system_prompt_col,
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
        system_prompt_col: Optional[str] = None,
    ) -> List[InternalsInstance]:
        instances: List[InternalsInstance] = []
        all_cols = (
            set(dataset.column_names)
            if hasattr(dataset, "column_names")
            else set(next(iter(dataset)).keys())
        )
        exclude = {text_col}
        if spans_col:
            exclude.add(spans_col)
        if system_prompt_col:
            exclude.add(system_prompt_col)
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

            # Per-row system prompt from dataset column
            row_system_prompt: Optional[str] = None
            if system_prompt_col and system_prompt_col in row:
                val = row[system_prompt_col]
                if val:
                    row_system_prompt = str(val)

            instances.append(InternalsInstance(
                text=text,
                properties=props,
                spans=merged_spans if merged_spans else None,
                system_prompt=row_system_prompt,
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
        system_prompt_col: Optional[str] = None,
    ) -> "InternalsDataset":
        """Backward-compatible classmethod constructor."""
        return cls(
            dataset,
            text_col=text_col,
            property_cols=property_cols,
            spans=spans,
            spans_col=spans_col,
            system_prompt_col=system_prompt_col,
        )

    def run(
        self,
        model: Any,
        tokenizer: Any,
        generate_kwargs: Optional[Dict[str, Any]] = None,
        finalize_timeout: float = 60.0,
        verbose: bool = True,
        system_prompt: Optional[str] = None,
        thinking_end_token: Optional[str] = None,
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
        system_prompt : str, optional
            Default system prompt applied to every instance that does not
            already have its own ``instance.system_prompt`` set.

            The input is formatted via ``tokenizer.apply_chat_template``
            (with a plain-concatenation fallback)::

                dataset.run(
                    model, tokenizer,
                    system_prompt="You are a helpful assistant.",
                    generate_kwargs={"max_new_tokens": 50},
                )

            To use *different* system prompts per instance, set
            ``InternalsInstance.system_prompt`` directly.  Per-instance
            system prompts always take priority over this argument.
        thinking_end_token : str, optional
            Token string marking the end of the thinking phase for reasoning
            models (e.g. ``"</think>"`` for DeepSeek-R1 / Qwen3).

            After generation lamina scans the output token IDs for the *last*
            occurrence of this token and stores the hidden state at that
            position in :attr:`InternalsRecord.thinking_hidden_state`
            (shape ``(num_layers, hidden_dim)``) and the output-sequence
            position in :attr:`InternalsRecord.thinking_end_token_pos`::

                records = dataset.run(
                    model, tokenizer,
                    thinking_end_token="</think>",
                    generate_kwargs={"max_new_tokens": 512},
                )
                hs = records[0].thinking_hidden_state   # (num_layers, hidden)
                pos = records[0].thinking_end_token_pos  # int

            Per-instance ``thinking_end_token`` always takes priority.
            If the token is not present in the generated output,
            ``thinking_hidden_state`` is ``None`` and a warning is issued.
        """
        import lamina
        from lamina.extractors.hf.extractor import run_forward
        from lamina.extractors.hf import extractor as _ext

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

            # ── Effective system prompt (instance overrides run-level) ─────────
            effective_sp: Optional[str] = instance.system_prompt or system_prompt

            # ── Effective thinking token (instance overrides run-level) ────────
            effective_tt: Optional[str] = (
                instance.thinking_end_token or thinking_end_token
            )

            # ── Encode ────────────────────────────────────────────────────────
            # text_for_spans is the string actually passed to the tokenizer;
            # used for TextSpan resolution so offsets match the real input.
            if isinstance(instance.text, str):
                if effective_sp is not None:
                    text_for_spans = _apply_system_prompt(
                        instance.text, effective_sp, tokenizer
                    )
                else:
                    text_for_spans = instance.text

                enc       = tokenizer(text_for_spans, return_tensors="pt")
                enc       = {k: v.to(model.device) for k, v in enc.items()}
                input_ids = enc["input_ids"]
            else:
                import torch
                text_for_spans = None   # pre-tokenised — no TextSpan support
                input_ids = torch.tensor([instance.text], device=model.device)
                enc = {"input_ids": input_ids}

            # ── Resolve text spans ────────────────────────────────────────────
            resolved_spans: Optional[Dict[str, SpanSpec]] = None
            if instance.spans and text_for_spans is not None:
                resolved_spans = _resolve_text_spans(
                    text_for_spans, instance.spans, tokenizer
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

            # ── Resolve thinking end token ID ────────────────────────────────
            tt_id: Optional[int] = None
            if effective_tt is not None and use_generate:
                tt_ids = tokenizer.encode(effective_tt, add_special_tokens=False)
                if not tt_ids:
                    warnings.warn(
                        f"[lamina] thinking_end_token {effective_tt!r} encodes "
                        "to an empty sequence — skipping thinking extraction.",
                        stacklevel=2,
                    )
                    effective_tt = None  # disable for this instance
                else:
                    tt_id = tt_ids[-1]  # anchor on the last sub-token

            # ── Inference ─────────────────────────────────────────────────────
            # Pass thinking token ID to the extractor so the in-worker detector
            # can capture the hidden state without storing all output hs.
            gen_output = None
            if use_generate:
                if tt_id is not None:
                    _ext._next_thinking_end_token_id = tt_id
                kwargs = dict(gkw)
                if "attention_mask" in enc:
                    kwargs.setdefault("attention_mask", enc["attention_mask"])
                if effective_tt is not None:
                    kwargs.setdefault("return_dict_in_generate", True)
                try:
                    gen_output = model.generate(input_ids, **kwargs)
                finally:
                    _ext._next_thinking_end_token_id = None  # always reset
            else:
                run_forward(model, **enc)

            # ── Wait for background worker ────────────────────────────────────
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

            # ── Thinking-token hidden state ───────────────────────────────────
            thinking_hs:  Optional[np.ndarray] = None
            thinking_pos: Optional[int]        = None

            if effective_tt is not None and tt_id is not None and use_generate:
                # Fast path: in-worker detection stored result on the run
                if run.thinking_end_hidden_state is not None:
                    thinking_hs  = run.thinking_end_hidden_state
                    thinking_pos = run.thinking_end_token_pos

                # Fallback: scan output_hidden_states (when available)
                elif run.output_hidden_states is not None and gen_output is not None:
                    if hasattr(gen_output, "sequences"):
                        full_ids = gen_output.sequences[0].cpu().numpy()
                    else:
                        full_ids = gen_output[0].cpu().numpy()
                    prompt_len   = int(input_ids.shape[-1])
                    generated_ids = full_ids[prompt_len:]
                    thinking_hs, thinking_pos = _extract_thinking_hidden_state(
                        run, generated_ids, tt_id
                    )
                    if thinking_hs is None:
                        warnings.warn(
                            f"[lamina] thinking_end_token {effective_tt!r} "
                            f"(token_id={tt_id}) was not found in the generated "
                            "output — thinking_hidden_state will be None.",
                            stacklevel=2,
                        )

                else:
                    warnings.warn(
                        "[lamina] thinking_end_token set but neither in-worker "
                        "detection succeeded nor output_hidden_states are available."
                        " Set extract_logits=True (needed for in-worker detection)"
                        " or extract_output_hidden_states=True (post-hoc fallback).",
                        stacklevel=2,
                    )

            # ── Span averages ─────────────────────────────────────────────────
            span_means = _compute_span_means(run, resolved_spans)
            records.append(InternalsRecord(
                instance=instance,
                run=run,
                resolved_spans=resolved_spans,
                span_hidden_states_mean=span_means,
                thinking_hidden_state=thinking_hs,
                thinking_end_token_pos=thinking_pos,
            ))

        if verbose:
            print(f"  Extraction complete — {len(records)} records.")

        return records
