"""
Tests for InternalsDataset, InternalsRecord, SpanSpec, TextSpan,
_resolve_text_spans, dump, load, to_hf_dataset, and from_hf_dataset.

All tests run without a GPU or a real model.
"""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from typing import Any, Dict, List

import numpy as np
import pytest

from internals_extraction._dataset import (
    SpanSpec,
    TextSpan,
    SpanResolutionError,
    InternalsInstance,
    InternalsRecord,
    InternalsDataset,
    _compute_span_means,
    _resolve_text_spans,
    _normalise_span,
)
from internals_extraction._dump import dump, load, to_hf_dataset
from internals_extraction._store import InternalsRun


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_run(
    input_len: int = 4,
    hidden: int = 8,
    num_layers: int = 2,
    num_output_tokens: int = 2,
    vocab: int = 20,
) -> InternalsRun:
    """Build a finalized InternalsRun with synthetic numpy arrays."""
    run   = InternalsRun(run_id=str(uuid.uuid4()), input_len=input_len)
    batch = 1

    run.input_hidden_states = [
        np.random.randn(batch, input_len, hidden).astype(np.float32)
        for _ in range(num_layers + 1)
    ]
    run.output_hidden_states = [
        np.random.randn(batch, num_output_tokens, hidden).astype(np.float32)
        for _ in range(num_layers + 1)
    ]
    run.input_hidden_states_mean = np.stack(
        [hs.mean(axis=1) for hs in run.input_hidden_states], axis=0
    )
    run.output_hidden_states_mean = np.stack(
        [hs.mean(axis=1) for hs in run.output_hidden_states], axis=0
    )
    run.logits     = np.random.randn(num_output_tokens, batch, vocab).astype(np.float32)
    run.attentions = None
    run.logit_lens = None
    run._finalized = True
    return run


class FakeEncoding:
    """Minimal offset-mapping encoding stub."""
    def __init__(self, text: str, tokens: List[str], offsets: List):
        self._data = {"offset_mapping": offsets}
        self.input_ids = list(range(len(tokens)))

    def __getitem__(self, key):
        return self._data[key]


class FakeTokenizer:
    """
    Minimal tokenizer that splits on spaces and returns exact character offsets.
    Supports return_offsets_mapping=True.
    """
    is_fast = True

    def __call__(self, text, return_offsets_mapping=False, return_tensors=None):
        # Very simple whitespace tokenizer
        tokens = []
        offsets = []
        pos = 0
        for word in text.split(" "):
            start = text.index(word, pos)
            end   = start + len(word)
            tokens.append(word)
            offsets.append((start, end))
            pos = end

        enc = {"input_ids": list(range(len(tokens)))}
        if return_offsets_mapping:
            enc["offset_mapping"] = offsets
        return enc

    def convert_ids_to_tokens(self, ids):
        return [str(i) for i in ids]


# ---------------------------------------------------------------------------
# SpanSpec
# ---------------------------------------------------------------------------

class TestSpanSpec:
    def test_basic(self):
        s = SpanSpec(start=1, end=4)
        assert s.length == 3
        assert s.as_slice() == slice(1, 4)

    def test_zero_length(self):
        s = SpanSpec(start=3, end=3)
        assert s.length == 0

    def test_invalid_positive(self):
        with pytest.raises(ValueError):
            SpanSpec(start=5, end=2)

    def test_negative_end_allowed(self):
        s = SpanSpec(start=1, end=-1)
        assert s.start == 1 and s.end == -1


# ---------------------------------------------------------------------------
# TextSpan
# ---------------------------------------------------------------------------

class TestTextSpan:
    def test_basic(self):
        ts = TextSpan("The cat")
        assert ts.text == "The cat"
        assert ts.strip is True
        assert ts.label is None
        assert ts.occurrence == 0

    def test_label(self):
        ts = TextSpan("The cat", label="subject")
        assert ts.label == "subject"

    def test_occurrence(self):
        ts = TextSpan("the", occurrence=2)
        assert ts.occurrence == 2

    def test_occurrence_negative_raises(self):
        with pytest.raises(ValueError, match="occurrence"):
            TextSpan("the", occurrence=-1)

    def test_strip_false(self):
        ts = TextSpan(" hello ", strip=False)
        assert ts.strip is False


# ---------------------------------------------------------------------------
# _normalise_span
# ---------------------------------------------------------------------------

class TestNormaliseSpan:
    def test_str_becomes_textspan(self):
        result = _normalise_span("hello world")
        assert isinstance(result, TextSpan)
        assert result.text == "hello world"

    def test_tuple_becomes_spanspec(self):
        result = _normalise_span((2, 5))
        assert isinstance(result, SpanSpec)
        assert result.start == 2 and result.end == 5

    def test_spanspec_passthrough(self):
        s = SpanSpec(1, 3)
        assert _normalise_span(s) is s

    def test_textspan_passthrough(self):
        ts = TextSpan("foo")
        assert _normalise_span(ts) is ts

    def test_invalid_type(self):
        with pytest.raises(TypeError):
            _normalise_span(42)


# ---------------------------------------------------------------------------
# InternalsInstance normalisation
# ---------------------------------------------------------------------------

class TestInternalsInstance:
    def test_str_spans_normalised_to_textspan(self):
        inst = InternalsInstance(
            text="The cat sat on the mat.",
            properties={"label": 1},
            spans={"subject": "The cat", "predicate": "sat"},
        )
        assert isinstance(inst.spans["subject"], TextSpan)
        assert isinstance(inst.spans["predicate"], TextSpan)
        assert inst.spans["subject"].text == "The cat"

    def test_tuple_spans_normalised(self):
        inst = InternalsInstance(
            text="hello world",
            spans={"a": (0, 2), "b": (2, 4)},
        )
        assert isinstance(inst.spans["a"], SpanSpec)
        assert inst.spans["b"].end == 4

    def test_spanspec_passthrough(self):
        inst = InternalsInstance(text="hi", spans={"tok": SpanSpec(1, 3)})
        assert inst.spans["tok"].start == 1

    def test_no_spans(self):
        inst = InternalsInstance(text="hello", properties={"x": 1})
        assert inst.spans is None

    def test_mixed_spans(self):
        inst = InternalsInstance(
            text="The cat sat.",
            spans={
                "by_text":  "The cat",
                "by_tuple": (2, 3),
                "by_spec":  SpanSpec(3, 4),
                "by_ts":    TextSpan("sat"),
            },
        )
        assert isinstance(inst.spans["by_text"],  TextSpan)
        assert isinstance(inst.spans["by_tuple"], SpanSpec)
        assert isinstance(inst.spans["by_spec"],  SpanSpec)
        assert isinstance(inst.spans["by_ts"],    TextSpan)

    def test_list_form_uses_labels_as_keys(self):
        inst = InternalsInstance(
            text="The cat sat on the mat.",
            spans=[
                TextSpan("The cat",        label="subject"),
                TextSpan("sat on the mat", label="predicate"),
            ],
        )
        assert set(inst.spans.keys()) == {"subject", "predicate"}
        assert isinstance(inst.spans["subject"], TextSpan)
        assert inst.spans["subject"].text == "The cat"

    def test_list_form_missing_label_raises(self):
        with pytest.raises(ValueError, match="no label"):
            InternalsInstance(
                text="hello world",
                spans=[TextSpan("hello")],  # no label
            )

    def test_list_form_duplicate_label_raises(self):
        with pytest.raises(ValueError, match="Duplicate"):
            InternalsInstance(
                text="hello world",
                spans=[
                    TextSpan("hello", label="a"),
                    TextSpan("world", label="a"),
                ],
            )

    def test_list_form_non_textspan_raises(self):
        with pytest.raises(TypeError, match="TextSpan"):
            InternalsInstance(
                text="hello",
                spans=[SpanSpec(0, 1)],  # SpanSpec has no label
            )


# ---------------------------------------------------------------------------
# _resolve_text_spans
# ---------------------------------------------------------------------------

class TestResolveTextSpans:
    """Tests for offset-mapping based TextSpan → SpanSpec resolution."""

    TEXT = "The cat sat on the mat"
    # Tokens (space-split): ["The", "cat", "sat", "on", "the", "mat"]
    # Indices:               0       1      2      3    4      5

    def test_single_textspan(self):
        spans = {"subject": TextSpan("The cat")}
        resolved = _resolve_text_spans(self.TEXT, spans, FakeTokenizer())
        assert isinstance(resolved["subject"], SpanSpec)
        assert resolved["subject"].start == 0
        assert resolved["subject"].end   == 2    # "The"=0, "cat"=1 → end=2

    def test_multi_token_span(self):
        spans = {"pred": TextSpan("sat on the mat")}
        resolved = _resolve_text_spans(self.TEXT, spans, FakeTokenizer())
        s = resolved["pred"]
        assert s.start == 2
        assert s.end   == 6

    def test_spanspec_unchanged(self):
        spans = {"fixed": SpanSpec(1, 3)}
        resolved = _resolve_text_spans(self.TEXT, spans, FakeTokenizer())
        assert resolved["fixed"].start == 1
        assert resolved["fixed"].end   == 3

    def test_mixed_span_types(self):
        spans = {
            "text_span": TextSpan("The cat"),
            "token_span": SpanSpec(4, 6),
        }
        resolved = _resolve_text_spans(self.TEXT, spans, FakeTokenizer())
        assert isinstance(resolved["text_span"],  SpanSpec)
        assert isinstance(resolved["token_span"], SpanSpec)
        assert resolved["text_span"].start == 0

    def test_substring_not_found(self):
        spans = {"missing": TextSpan("missing text")}
        with pytest.raises(SpanResolutionError, match="not found"):
            _resolve_text_spans(self.TEXT, spans, FakeTokenizer())

    def test_no_textspan_skips_tokenization(self):
        """If only SpanSpec values, the tokenizer is never called."""
        class BrokenTokenizer:
            def __call__(self, *a, **kw):
                raise RuntimeError("should not be called")

        spans = {"s": SpanSpec(0, 1)}
        resolved = _resolve_text_spans(self.TEXT, spans, BrokenTokenizer())
        assert resolved["s"].start == 0

    def test_strip_whitespace(self):
        """Leading/trailing space in TextSpan.text should be ignored."""
        spans = {"s": TextSpan("  The cat  ", strip=True)}
        resolved = _resolve_text_spans(self.TEXT, spans, FakeTokenizer())
        assert resolved["s"].start == 0

    def test_strip_false_exact_match(self):
        """With strip=False, leading space must be present in the text."""
        spans = {"s": TextSpan(" cat", strip=False)}
        resolved = _resolve_text_spans(self.TEXT, spans, FakeTokenizer())
        assert resolved["s"].start == 1

    # ── occurrence tests ──────────────────────────────────────────────────────
    # TEXT = "The cat sat on the mat"
    # "the" appears at index 0 ("The") and index 4 ("the") — case-sensitive so
    # we test with "at" which appears in "cat"(1) and "mat"(5).
    # But FakeTokenizer is word-split, so substring within word won't map.
    # Use whole words instead: "on" appears once; "the" / "The" are different.
    # Use a text with a repeated whole word.

    TEXT_REPEAT = "the cat and the dog and the bird"
    # Tokens: ["the","cat","and","the","dog","and","the","bird"]
    # "the" at token indices 0, 3, 6

    def test_occurrence_0_is_first(self):
        spans = {"t": TextSpan("the", occurrence=0)}
        resolved = _resolve_text_spans(self.TEXT_REPEAT, spans, FakeTokenizer())
        assert resolved["t"].start == 0
        assert resolved["t"].end   == 1

    def test_occurrence_1_is_second(self):
        spans = {"t": TextSpan("the", occurrence=1)}
        resolved = _resolve_text_spans(self.TEXT_REPEAT, spans, FakeTokenizer())
        assert resolved["t"].start == 3
        assert resolved["t"].end   == 4

    def test_occurrence_2_is_third(self):
        spans = {"t": TextSpan("the", occurrence=2)}
        resolved = _resolve_text_spans(self.TEXT_REPEAT, spans, FakeTokenizer())
        assert resolved["t"].start == 6
        assert resolved["t"].end   == 7

    def test_occurrence_out_of_range_raises(self):
        spans = {"t": TextSpan("the", occurrence=5)}
        with pytest.raises(SpanResolutionError, match="occurrence=5"):
            _resolve_text_spans(self.TEXT_REPEAT, spans, FakeTokenizer())

    def test_multiple_spans_different_occurrences(self):
        """Two spans targeting different occurrences of the same word."""
        spans = {
            "the_first":  TextSpan("the", occurrence=0),
            "the_second": TextSpan("the", occurrence=1),
        }
        resolved = _resolve_text_spans(self.TEXT_REPEAT, spans, FakeTokenizer())
        assert resolved["the_first"].start  == 0
        assert resolved["the_second"].start == 3


# ---------------------------------------------------------------------------
# _compute_span_means  (already covered but re-run with resolved SpanSpec)
# ---------------------------------------------------------------------------

class TestComputeSpanMeans:
    def test_basic_span(self):
        run   = _make_run(input_len=6, hidden=8, num_layers=2)
        spans = {"subject": SpanSpec(1, 4)}
        result = _compute_span_means(run, spans)
        assert result is not None
        assert result["subject"].shape == (3, 8)   # num_layers+1=3

    def test_span_value_correctness(self):
        run = _make_run(input_len=4, hidden=4, num_layers=1)
        run.input_hidden_states[0][:] = 5.0
        result = _compute_span_means(run, {"all": SpanSpec(0, 4)})
        np.testing.assert_allclose(result["all"][0], 5.0, rtol=1e-5)

    def test_empty_span_returns_zeros(self):
        run    = _make_run(input_len=4, hidden=8, num_layers=1)
        result = _compute_span_means(run, {"e": SpanSpec(2, 2)})
        np.testing.assert_array_equal(result["e"][0], 0.0)

    def test_negative_end_resolved(self):
        run    = _make_run(input_len=6, hidden=4, num_layers=1)
        result = _compute_span_means(run, {"mid": SpanSpec(1, -1)})
        assert result["mid"].shape == (2, 4)

    def test_no_spans_returns_none(self):
        assert _compute_span_means(_make_run(), None) is None

    def test_no_hidden_states_returns_none(self):
        run = _make_run()
        run.input_hidden_states = None
        assert _compute_span_means(run, {"s": SpanSpec(0, 1)}) is None

    def test_multiple_spans(self):
        run  = _make_run(input_len=8, hidden=4, num_layers=2)
        result = _compute_span_means(run, {
            "a": SpanSpec(0, 2),
            "b": SpanSpec(2, 5),
            "c": SpanSpec(5, 8),
        })
        assert set(result.keys()) == {"a", "b", "c"}
        for arr in result.values():
            assert arr.shape == (3, 4)


# ---------------------------------------------------------------------------
# InternalsRecord
# ---------------------------------------------------------------------------

class TestInternalsRecord:
    def test_properties_shortcut(self):
        inst   = InternalsInstance(text="x", properties={"label": 7})
        record = InternalsRecord(instance=inst, run=_make_run())
        assert record.properties == {"label": 7}

    def test_spans_shortcut(self):
        inst   = InternalsInstance(text="x", spans={"s": SpanSpec(0, 1)})
        record = InternalsRecord(instance=inst, run=_make_run())
        assert "s" in record.spans

    def test_resolved_spans_stored(self):
        run    = _make_run(input_len=4, hidden=8, num_layers=2)
        inst   = InternalsInstance(text="x", spans={"tok": SpanSpec(1, 3)})
        resolved = {"tok": SpanSpec(1, 3)}
        span_means = _compute_span_means(run, resolved)
        record = InternalsRecord(
            instance=inst, run=run,
            resolved_spans=resolved,
            span_hidden_states_mean=span_means,
        )
        assert record.resolved_spans["tok"].start == 1
        assert record.span_hidden_states_mean["tok"].shape == (3, 8)


# ---------------------------------------------------------------------------
# dump / load round-trip  (with resolved_spans in metadata)
# ---------------------------------------------------------------------------

class TestDumpLoad:

    def _make_records(self, n=3, input_len=4, hidden=8,
                      num_layers=2, vocab=20) -> List[InternalsRecord]:
        records = []
        for i in range(n):
            run  = _make_run(input_len=input_len, hidden=hidden,
                             num_layers=num_layers, vocab=vocab)
            inst = InternalsInstance(
                text=f"sentence {i}",
                properties={"label": i % 2, "id": i, "task": "cls"},
                spans={"prefix": SpanSpec(0, 2), "suffix": SpanSpec(2, input_len)},
            )
            resolved   = {"prefix": SpanSpec(0, 2), "suffix": SpanSpec(2, input_len)}
            span_means = _compute_span_means(run, resolved)
            records.append(InternalsRecord(
                instance=inst, run=run,
                resolved_spans=resolved,
                span_hidden_states_mean=span_means,
            ))
        return records

    def test_files_created(self):
        records = self._make_records(3)
        with tempfile.TemporaryDirectory() as outdir:
            dump(records, outdir)
            files = os.listdir(outdir)
            assert "metadata.jsonl" in files
            for i in range(3):
                assert f"{i:05d}.npz" in files

    def test_metadata_span_boundaries(self):
        """resolved_spans written as {name: {start, end}} dicts."""
        records = self._make_records(1)
        with tempfile.TemporaryDirectory() as outdir:
            dump(records, outdir)
            with open(os.path.join(outdir, "metadata.jsonl")) as f:
                meta = json.loads(f.readline())
            assert meta["spans"]["prefix"] == {"start": 0, "end": 2}
            assert meta["spans"]["suffix"] == {"start": 2, "end": 4}

    def test_arrays_shape(self):
        records = self._make_records(1, input_len=4, hidden=8, num_layers=2)
        with tempfile.TemporaryDirectory() as outdir:
            dump(records, outdir)
            arrays_list, _ = load(outdir)
            a = arrays_list[0]
            assert a["input_hidden_states_mean"].shape  == (3, 8)
            assert a["output_hidden_states_mean"].shape == (3, 8)
            assert a["logits"].shape                    == (2, 20)
            assert a["span_prefix"].shape               == (3, 8)
            assert a["span_suffix"].shape               == (3, 8)

    def test_properties_round_trip(self):
        records = self._make_records(2)
        with tempfile.TemporaryDirectory() as outdir:
            dump(records, outdir)
            _, metadata_list = load(outdir)
            for i, meta in enumerate(metadata_list):
                assert meta["properties"]["label"] == i % 2
                assert meta["properties"]["task"]  == "cls"

    def test_full_hidden_states_optional(self):
        records = self._make_records(1, input_len=4, hidden=8, num_layers=2)
        with tempfile.TemporaryDirectory() as outdir:
            dump(records, outdir, save_full_hidden_states=True)
            arrays_list, _ = load(outdir)
            a = arrays_list[0]
            assert a["input_hidden_states"].shape  == (3, 4, 8)
            assert a["output_hidden_states"].shape == (3, 2, 8)

    def test_no_spans_record(self):
        run    = _make_run(input_len=4)
        inst   = InternalsInstance(text="x", properties={"label": 0})
        record = InternalsRecord(instance=inst, run=run)
        with tempfile.TemporaryDirectory() as outdir:
            dump([record], outdir)
            arrays_list, metadata_list = load(outdir)
            assert metadata_list[0]["spans"] == {}
            assert not any(k.startswith("span_") for k in arrays_list[0])

    def test_load_count(self):
        records = self._make_records(5)
        with tempfile.TemporaryDirectory() as outdir:
            dump(records, outdir)
            arrays_list, metadata_list = load(outdir)
            assert len(arrays_list)   == 5
            assert len(metadata_list) == 5


# ---------------------------------------------------------------------------
# to_hf_dataset
# ---------------------------------------------------------------------------

class TestToHfDataset:

    def _make_records(self, n=3) -> List[InternalsRecord]:
        records = []
        for i in range(n):
            run  = _make_run(input_len=4, hidden=8, num_layers=2)
            inst = InternalsInstance(
                text=f"sentence {i}",
                properties={"label": i % 2, "id": i},
                spans={"prefix": SpanSpec(0, 2)},
            )
            resolved   = {"prefix": SpanSpec(0, 2)}
            span_means = _compute_span_means(run, resolved)
            records.append(InternalsRecord(
                instance=inst, run=run,
                resolved_spans=resolved,
                span_hidden_states_mean=span_means,
            ))
        return records

    def _try_import(self):
        pytest.importorskip("datasets", reason="datasets not installed")

    def test_returns_hf_dataset(self):
        self._try_import()
        from datasets import Dataset
        records = self._make_records(3)
        ds = to_hf_dataset(records)
        assert isinstance(ds, Dataset)

    def test_length(self):
        self._try_import()
        records = self._make_records(4)
        ds = to_hf_dataset(records)
        assert len(ds) == 4

    def test_property_columns_present(self):
        self._try_import()
        records = self._make_records(2)
        ds = to_hf_dataset(records)
        assert "label" in ds.column_names
        assert "id"    in ds.column_names

    def test_array_columns_present(self):
        self._try_import()
        records = self._make_records(2)
        ds = to_hf_dataset(records)
        assert "input_hidden_states_mean"  in ds.column_names
        assert "output_hidden_states_mean" in ds.column_names
        assert "span_prefix"               in ds.column_names

    def test_span_boundary_columns(self):
        self._try_import()
        records = self._make_records(2)
        ds = to_hf_dataset(records)
        assert "span_prefix_start" in ds.column_names
        assert "span_prefix_end"   in ds.column_names
        assert ds[0]["span_prefix_start"] == 0
        assert ds[0]["span_prefix_end"]   == 2

    def test_array_shape_preserved(self):
        self._try_import()
        records = self._make_records(1)
        ds = to_hf_dataset(records)
        arr = np.array(ds[0]["input_hidden_states_mean"])
        # (num_layers+1=3, hidden=8)
        assert arr.shape == (3, 8)

    def test_run_id_present(self):
        self._try_import()
        records = self._make_records(2)
        ds = to_hf_dataset(records)
        assert "run_id" in ds.column_names
        assert isinstance(ds[0]["run_id"], str)

    def test_missing_datasets_raises(self, monkeypatch):
        """If datasets is not installed, a clear ImportError is raised."""
        import builtins
        real_import = builtins.__import__

        def _block(name, *args, **kwargs):
            if name == "datasets":
                raise ImportError("mocked missing")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block)
        with pytest.raises(ImportError, match="datasets"):
            to_hf_dataset(self._make_records(1))


# ---------------------------------------------------------------------------
# from_hf_dataset
# ---------------------------------------------------------------------------

class TestFromHfDataset:

    def _fake_hf_ds(self, n=4):
        """Minimal dict-list that mimics a datasets.Dataset."""
        class FakeDS:
            column_names = ["text", "label", "idx"]

            def __init__(self, rows):
                self._rows = rows

            def __iter__(self):
                return iter(self._rows)

            def __len__(self):
                return len(self._rows)

        return FakeDS([
            {"text": f"sentence number {i}", "label": i % 2, "idx": i}
            for i in range(n)
        ])

    def test_basic(self):
        ds   = self._fake_hf_ds(4)
        idst = InternalsDataset.from_hf_dataset(ds, text_col="text")
        assert len(idst) == 4
        assert idst[0].text == "sentence number 0"

    def test_properties_extracted(self):
        ds   = self._fake_hf_ds(3)
        idst = InternalsDataset.from_hf_dataset(
            ds, text_col="text", property_cols=["label", "idx"]
        )
        assert idst[0].properties["label"] == 0
        assert idst[1].properties["idx"]   == 1

    def test_all_cols_as_properties_by_default(self):
        ds   = self._fake_hf_ds(2)
        idst = InternalsDataset.from_hf_dataset(ds, text_col="text")
        # label and idx should both appear as properties
        assert "label" in idst[0].properties
        assert "idx"   in idst[0].properties

    def test_global_spans_applied(self):
        ds   = self._fake_hf_ds(2)
        idst = InternalsDataset.from_hf_dataset(
            ds,
            text_col="text",
            spans={"first_word": SpanSpec(0, 1)},
        )
        assert "first_word" in idst[0].spans
        assert "first_word" in idst[1].spans

    def test_global_textspan_applied(self):
        ds   = self._fake_hf_ds(2)
        idst = InternalsDataset.from_hf_dataset(
            ds,
            text_col="text",
            spans={"word": TextSpan("sentence")},
        )
        assert isinstance(idst[0].spans["word"], TextSpan)

    def test_per_row_spans_col(self):
        class FakeDS:
            column_names = ["text", "label", "my_spans"]
            _rows = [
                {"text": "hello world", "label": 0,
                 "my_spans": {"subject": "hello", "object": "world"}},
                {"text": "foo bar baz", "label": 1,
                 "my_spans": {"noun": [0, 1]}},
            ]
            def __iter__(self): return iter(self._rows)
            def __len__(self):  return len(self._rows)

        idst = InternalsDataset.from_hf_dataset(
            FakeDS(), text_col="text", spans_col="my_spans"
        )
        # Row 0: text-based spans
        assert isinstance(idst[0].spans["subject"], TextSpan)
        assert isinstance(idst[0].spans["object"],  TextSpan)
        # Row 1: token-index list
        assert isinstance(idst[1].spans["noun"], SpanSpec)
        assert idst[1].spans["noun"].start == 0
        assert idst[1].spans["noun"].end   == 1

    def test_per_row_spans_dict_format(self):
        class FakeDS:
            column_names = ["text", "spans"]
            _rows = [
                {"text": "hello world",
                 "spans": {"tok": {"start": 1, "end": 2}}},
            ]
            def __iter__(self): return iter(self._rows)
            def __len__(self):  return len(self._rows)

        idst = InternalsDataset.from_hf_dataset(
            FakeDS(), text_col="text", spans_col="spans"
        )
        assert isinstance(idst[0].spans["tok"], SpanSpec)
        assert idst[0].spans["tok"].start == 1
        assert idst[0].spans["tok"].end   == 2
