"""
Tests for all supported HuggingFace AutoModel categories.

Each test class covers one model family:

  TestModelCanGenerate         — _model_can_generate() detection logic
  TestEncoderDecoderFinalize   — _finalize() with encoder-decoder step data
  TestDecoderOnlyPipeline      — fake GPT-style model via generate() + hook
  TestEncoderDecoderPipeline   — fake T5-style model via generate() + hook
  TestEncoderOnlyPipeline      — fake BERT-style model via run_forward()
  TestSequenceClassifierPipeline — 2-D logits, non-generative
  TestTokenClassifierPipeline  — 3-D logits (batch, seq, labels), non-generative
  TestQAModelPipeline          — start_logits / end_logits, non-generative
  TestMaskedLMPipeline         — full-sequence vocab logits, non-generative

All tests run without a GPU or a real transformers checkpoint.
FakeModule implements nn.Module-like register_forward_hook so the real
patch mechanism (register_forward_hook inside _patched_generate /
run_forward) is exercised end-to-end.
"""
from __future__ import annotations

import time
import types
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fake tensor (no torch required)
# ---------------------------------------------------------------------------

class FakeTensor:
    """CPU numpy-backed tensor mimic used throughout all tests."""

    def __init__(self, array: np.ndarray):
        self._data = array.astype(np.float32)

    @property
    def shape(self):
        return self._data.shape

    def detach(self):  return self
    def cpu(self):     return self
    def float(self):   return self
    def numpy(self):   return self._data

    def __getitem__(self, item):
        return FakeTensor(self._data[item])


def _ft(arr) -> FakeTensor:
    return FakeTensor(np.array(arr, dtype=np.float32))


# ---------------------------------------------------------------------------
# Fake nn.Module-like base with register_forward_hook
# ---------------------------------------------------------------------------

class _HookHandle:
    def __init__(self, registry: dict, key: int):
        self._registry = registry
        self._key = key

    def remove(self):
        self._registry.pop(self._key, None)


class FakeModule:
    """
    Minimal nn.Module replacement.

    Supports ``register_forward_hook`` so that ``_patched_generate`` and
    ``run_forward`` work correctly without requiring PyTorch to be installed.
    ``__call__`` fires registered hooks after ``forward()`` returns, exactly
    as PyTorch's ``nn.Module.__call__`` does.
    """

    def __init__(self):
        self._forward_hooks: Dict[int, Any] = {}
        self._hook_counter = 0
        self.device = types.SimpleNamespace(__str__=lambda s: "cpu")

    def register_forward_hook(self, fn):
        key = self._hook_counter
        self._hook_counter += 1
        self._forward_hooks[key] = fn
        return _HookHandle(self._forward_hooks, key)

    def __call__(self, *args, **kwargs):
        output = self.forward(*args, **kwargs)
        for hook_fn in list(self._forward_hooks.values()):
            hook_fn(self, args, output)
        return output

    def forward(self, *args, **kwargs):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Named output types
# (class names matter for logit_mode detection in _patch.py)
# ---------------------------------------------------------------------------

def _output_cls(cls_name: str, **fields):
    """Create an output object whose __class__.__name__ == cls_name."""
    cls = type(cls_name, (), {})
    obj = object.__new__(cls)
    for k, v in fields.items():
        setattr(obj, k, v)
    # Ensure missing attributes return None
    original_getattr = cls.__getattribute__
    def _safe(self, name):
        try:
            return original_getattr(self, name)
        except AttributeError:
            return None
    cls.__getattribute__ = _safe
    return obj


# ---------------------------------------------------------------------------
# Fake model configs
# ---------------------------------------------------------------------------

class FakeConfig:
    def __init__(self, is_encoder_decoder: bool = False, model_type: str = "custom"):
        self.is_encoder_decoder = is_encoder_decoder
        self.model_type = model_type


# ---------------------------------------------------------------------------
# Concrete fake models
# ---------------------------------------------------------------------------

BATCH, INPUT_LEN, HIDDEN, VOCAB, NUM_LAYERS, NUM_HEADS = 1, 5, 8, 20, 2, 2


def _make_hs(batch, seq, hidden, num_layers):
    """Return list of FakeTensor per layer (embedding + N blocks)."""
    return [
        FakeTensor(np.random.randn(batch, seq, hidden).astype(np.float32))
        for _ in range(num_layers + 1)
    ]


def _make_att(batch, heads, seq_q, seq_k, num_layers):
    return [
        FakeTensor(np.random.rand(batch, heads, seq_q, seq_k).astype(np.float32))
        for _ in range(num_layers)
    ]


class FakeDecoderOnlyModel(FakeModule):
    """
    GPT-2 style decoder-only model.
    Returns CausalLMOutputWithPast (class name contains 'CausalLM').
    """
    config = FakeConfig(is_encoder_decoder=False, model_type="gpt2")

    def can_generate(self):
        return True

    def forward(self, input_ids=None, output_hidden_states=False,
                output_attentions=False, return_dict=True, **kwargs):
        batch = input_ids.shape[0] if input_ids is not None else BATCH
        seq   = input_ids.shape[1] if input_ids is not None else INPUT_LEN
        return _output_cls(
            "CausalLMOutputWithPast",
            hidden_states=tuple(_make_hs(batch, seq, HIDDEN, NUM_LAYERS))
                          if output_hidden_states else None,
            attentions=tuple(_make_att(batch, NUM_HEADS, seq, seq, NUM_LAYERS))
                       if output_attentions else None,
            logits=FakeTensor(np.random.randn(batch, seq, VOCAB).astype(np.float32)),
            encoder_hidden_states=None,
            decoder_hidden_states=None,
        )


class FakeEncoderDecoderModel(FakeModule):
    """
    T5 style encoder-decoder model.
    Output has encoder_hidden_states + decoder_hidden_states (Seq2SeqLMOutput).
    The encoder runs once (step 0); decoder runs at every step.
    """
    config = FakeConfig(is_encoder_decoder=True, model_type="t5")

    def __init__(self):
        super().__init__()
        self._step_counter = 0  # tracks how many times forward was called

    def can_generate(self):
        return True

    def forward(self, input_ids=None, output_hidden_states=False,
                output_attentions=False, return_dict=True, **kwargs):
        batch   = input_ids.shape[0] if input_ids is not None else BATCH
        enc_len = input_ids.shape[1] if input_ids is not None else INPUT_LEN
        step    = self._step_counter
        self._step_counter += 1

        enc_hs = tuple(_make_hs(batch, enc_len, HIDDEN, NUM_LAYERS)) \
                 if (output_hidden_states and step == 0) else None
        dec_hs = tuple(_make_hs(batch, 1, HIDDEN, NUM_LAYERS)) \
                 if output_hidden_states else None

        return _output_cls(
            "Seq2SeqLMOutput",
            # encoder-decoder models use these separate attributes
            decoder_hidden_states=dec_hs,
            encoder_hidden_states=enc_hs,
            decoder_attentions=tuple(_make_att(batch, NUM_HEADS, 1, 1, NUM_LAYERS))
                               if output_attentions else None,
            encoder_attentions=tuple(_make_att(batch, NUM_HEADS, enc_len, enc_len, NUM_LAYERS))
                               if (output_attentions and step == 0) else None,
            logits=FakeTensor(np.random.randn(batch, 1, VOCAB).astype(np.float32)),
            # Seq2SeqLMOutput does NOT have plain hidden_states
            hidden_states=None,
            attentions=None,
        )


class FakeEncoderOnlyModel(FakeModule):
    """
    BERT base model (no head).
    Returns BaseModelOutput with hidden_states but no logits.
    """
    config = FakeConfig(is_encoder_decoder=False, model_type="bert")

    def can_generate(self):
        return False

    def forward(self, input_ids=None, attention_mask=None,
                output_hidden_states=False, output_attentions=False,
                return_dict=True, **kwargs):
        batch = input_ids.shape[0] if input_ids is not None else BATCH
        seq   = input_ids.shape[1] if input_ids is not None else INPUT_LEN
        return _output_cls(
            "BaseModelOutput",
            hidden_states=tuple(_make_hs(batch, seq, HIDDEN, NUM_LAYERS))
                          if output_hidden_states else None,
            attentions=tuple(_make_att(batch, NUM_HEADS, seq, seq, NUM_LAYERS))
                       if output_attentions else None,
            logits=None,
            encoder_hidden_states=None,
        )


class FakeSequenceClassifier(FakeModule):
    """
    BertForSequenceClassification style.
    Logits are 2-D: (batch, num_labels).
    """
    NUM_LABELS = 3
    config = FakeConfig(is_encoder_decoder=False, model_type="bert")

    def can_generate(self):
        return False

    def forward(self, input_ids=None, attention_mask=None,
                output_hidden_states=False, output_attentions=False,
                return_dict=True, **kwargs):
        batch = input_ids.shape[0] if input_ids is not None else BATCH
        seq   = input_ids.shape[1] if input_ids is not None else INPUT_LEN
        return _output_cls(
            "SequenceClassifierOutput",
            hidden_states=tuple(_make_hs(batch, seq, HIDDEN, NUM_LAYERS))
                          if output_hidden_states else None,
            attentions=None,
            logits=FakeTensor(np.random.randn(batch, self.NUM_LABELS).astype(np.float32)),
        )


class FakeTokenClassifier(FakeModule):
    """
    BertForTokenClassification style.
    Logits are 3-D: (batch, seq, num_labels).
    """
    NUM_LABELS = 9
    config = FakeConfig(is_encoder_decoder=False, model_type="bert")

    def can_generate(self):
        return False

    def forward(self, input_ids=None, attention_mask=None,
                output_hidden_states=False, output_attentions=False,
                return_dict=True, **kwargs):
        batch = input_ids.shape[0] if input_ids is not None else BATCH
        seq   = input_ids.shape[1] if input_ids is not None else INPUT_LEN
        return _output_cls(
            "TokenClassifierOutput",
            hidden_states=tuple(_make_hs(batch, seq, HIDDEN, NUM_LAYERS))
                          if output_hidden_states else None,
            attentions=None,
            logits=FakeTensor(
                np.random.randn(batch, seq, self.NUM_LABELS).astype(np.float32)
            ),
        )


class FakeQAModel(FakeModule):
    """
    BertForQuestionAnswering style.
    No combined logits — only start_logits and end_logits.
    """
    config = FakeConfig(is_encoder_decoder=False, model_type="bert")

    def can_generate(self):
        return False

    def forward(self, input_ids=None, attention_mask=None,
                output_hidden_states=False, output_attentions=False,
                return_dict=True, **kwargs):
        batch = input_ids.shape[0] if input_ids is not None else BATCH
        seq   = input_ids.shape[1] if input_ids is not None else INPUT_LEN
        return _output_cls(
            "QuestionAnsweringModelOutput",
            hidden_states=tuple(_make_hs(batch, seq, HIDDEN, NUM_LAYERS))
                          if output_hidden_states else None,
            attentions=None,
            logits=None,
            start_logits=FakeTensor(np.random.randn(batch, seq).astype(np.float32)),
            end_logits=FakeTensor(np.random.randn(batch, seq).astype(np.float32)),
        )


class FakeMaskedLMModel(FakeModule):
    """
    BertForMaskedLM style.
    Logits are 3-D (batch, seq, vocab) — full-sequence predictions.
    """
    config = FakeConfig(is_encoder_decoder=False, model_type="bert")

    def can_generate(self):
        return False

    def forward(self, input_ids=None, attention_mask=None,
                output_hidden_states=False, output_attentions=False,
                return_dict=True, **kwargs):
        batch = input_ids.shape[0] if input_ids is not None else BATCH
        seq   = input_ids.shape[1] if input_ids is not None else INPUT_LEN
        return _output_cls(
            "MaskedLMOutput",
            hidden_states=tuple(_make_hs(batch, seq, HIDDEN, NUM_LAYERS))
                          if output_hidden_states else None,
            attentions=None,
            logits=FakeTensor(
                np.random.randn(batch, seq, VOCAB).astype(np.float32)
            ),
        )


# ---------------------------------------------------------------------------
# Fake input_ids (shape-only, no real data needed)
# ---------------------------------------------------------------------------

class _FakeInputIds:
    """Minimal stand-in for a (batch, seq) tensor."""
    def __init__(self, batch=BATCH, seq=INPUT_LEN):
        self.shape = (batch, seq)

    def __getitem__(self, item):
        return _FakeInputIds(*[self.shape[i] for i in
                               ([item] if isinstance(item, int)
                                else range(len(self.shape)))])


def _fake_inputs(batch=BATCH, seq=INPUT_LEN):
    """Return {input_ids: _FakeInputIds} as used by run_forward / generate."""
    return {"input_ids": _FakeInputIds(batch, seq)}


# ---------------------------------------------------------------------------
# Plugin fixture — fresh singletons per test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def fresh_plugin(monkeypatch):
    import internals_extraction
    from internals_extraction._config import InternalsConfig
    from internals_extraction._store  import InternalsStore
    from internals_extraction._worker import BackgroundWorker
    from internals_extraction import _patch

    cfg    = InternalsConfig()
    store  = InternalsStore(maxlen=20)
    worker = BackgroundWorker(store, maxsize=0)
    _patch._initialise(cfg, store, worker)

    monkeypatch.setattr(internals_extraction, "config",  cfg)
    monkeypatch.setattr(internals_extraction, "_store",  store)
    monkeypatch.setattr(internals_extraction, "_worker", worker)

    yield cfg, store, worker
    worker.stop()


def _wait(run_id: str, timeout: float = 5.0):
    import internals_extraction
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = internals_extraction.get_run(run_id)
        if run is not None and run.is_finalized:
            return run
        time.sleep(0.02)
    pytest.fail(f"Run {run_id!r} not finalized within {timeout}s")


# ---------------------------------------------------------------------------
# Helper: simulate generate() loop without real transformers
# ---------------------------------------------------------------------------

def _fake_generate_loop(model, input_ids, max_new_tokens=2, **kwargs):
    """
    Minimal generate simulation: calls model() max_new_tokens+1 times.
    Step 0 passes full input_ids; steps 1..N pass a single-token input.
    Used as a monkeypatch for _patch._ORIGINAL_GENERATE.
    """
    # Pass through any output_hidden_states / output_attentions kwargs
    fwd_kw = {k: v for k, v in kwargs.items()
              if k in ("output_hidden_states", "output_attentions", "return_dict")}
    # Step 0: full prefix
    model(input_ids, **fwd_kw)
    # Steps 1..max_new_tokens: single new token (simulate KV-cache)
    single = _FakeInputIds(input_ids.shape[0], 1)
    for _ in range(max_new_tokens):
        model(single, **fwd_kw)
    return input_ids  # dummy return value


# ---------------------------------------------------------------------------
# 1. TestModelCanGenerate
# ---------------------------------------------------------------------------

class TestModelCanGenerate:
    """Tests for _model_can_generate() with every AutoModel category."""

    from internals_extraction._dataset import _model_can_generate as _mcg
    _model_can_generate = staticmethod(_mcg)

    def _model(self, cls_name, *, is_encoder_decoder=False, model_type="custom",
               can_generate_rv=None):
        cfg = FakeConfig(is_encoder_decoder=is_encoder_decoder,
                         model_type=model_type)
        methods = {}
        if can_generate_rv is not None:
            rv = can_generate_rv
            methods["can_generate"] = lambda self: rv
        cls = type(cls_name, (object,), methods)
        obj = object.__new__(cls)
        obj.config = cfg
        return obj

    # ── Generative (expect True) ─────────────────────────────────────��─────
    @pytest.mark.parametrize("cls_name", [
        "GPT2ForCausalLM",
        "LlamaForCausalLM",
        "MistralForCausalLM",
        "FalconForCausalLM",
    ])
    def test_causal_lm_variants(self, cls_name):
        assert self._model_can_generate(self._model(cls_name)) is True

    @pytest.mark.parametrize("cls_name", [
        "GPT2LMHeadModel",
        "GPTNeoXForCausalLM",
    ])
    def test_lm_head_model(self, cls_name):
        assert self._model_can_generate(self._model(cls_name)) is True

    @pytest.mark.parametrize("cls_name", [
        "T5ForSeq2SeqLM",
        "MBartForSeq2SeqLM",
    ])
    def test_seq2seq_lm(self, cls_name):
        assert self._model_can_generate(self._model(cls_name)) is True

    @pytest.mark.parametrize("cls_name", [
        "T5ForConditionalGeneration",
        "BartForConditionalGeneration",
        "PegasusForConditionalGeneration",
        "MarianMTModel",
    ])
    def test_conditional_generation(self, cls_name):
        assert self._model_can_generate(self._model(cls_name)) is True

    def test_speech_seq2seq(self):
        assert self._model_can_generate(
            self._model("WhisperForSpeechSeq2Seq")) is True

    def test_vision2seq(self):
        assert self._model_can_generate(
            self._model("BlipForVision2Seq")) is True

    # ── Non-generative (expect False) ─────────────────────────────────────
    @pytest.mark.parametrize("cls_name", [
        "BertForMaskedLM",
        "RobertaForMaskedLM",
        "AlbertForMaskedLM",
        "DistilBertForMaskedLM",
    ])
    def test_masked_lm(self, cls_name):
        assert self._model_can_generate(self._model(cls_name)) is False

    @pytest.mark.parametrize("cls_name", [
        "BertForSequenceClassification",
        "RobertaForSequenceClassification",
        "DistilBertForSequenceClassification",
        "DebertaForSequenceClassification",
    ])
    def test_sequence_classification(self, cls_name):
        assert self._model_can_generate(self._model(cls_name)) is False

    @pytest.mark.parametrize("cls_name", [
        "BertForTokenClassification",
        "RobertaForTokenClassification",
    ])
    def test_token_classification(self, cls_name):
        assert self._model_can_generate(self._model(cls_name)) is False

    @pytest.mark.parametrize("cls_name", [
        "BertForQuestionAnswering",
        "RobertaForQuestionAnswering",
        "DistilBertForQuestionAnswering",
    ])
    def test_question_answering(self, cls_name):
        assert self._model_can_generate(self._model(cls_name)) is False

    def test_multiple_choice(self):
        assert self._model_can_generate(
            self._model("BertForMultipleChoice")) is False

    def test_next_sentence_prediction(self):
        assert self._model_can_generate(
            self._model("BertForNextSentencePrediction")) is False

    @pytest.mark.parametrize("cls_name", [
        "ViTForImageClassification",
        "DeiTForImageClassification",
    ])
    def test_image_classification(self, cls_name):
        assert self._model_can_generate(self._model(cls_name)) is False

    def test_audio_classification(self):
        assert self._model_can_generate(
            self._model("Wav2Vec2ForAudioClassification")) is False

    def test_ctc(self):
        assert self._model_can_generate(
            self._model("Wav2Vec2ForCTC")) is False

    def test_feature_extraction(self):
        assert self._model_can_generate(
            self._model("BertForFeatureExtraction")) is False

    # ── Config-based detection ─────────────────────────────────────────────
    def test_encoder_decoder_config_flag(self):
        m = self._model("UnknownModel", is_encoder_decoder=True)
        assert self._model_can_generate(m) is True

    @pytest.mark.parametrize("mt", ["bert", "roberta", "distilbert",
                                     "albert", "deberta", "electra"])
    def test_encoder_only_model_type(self, mt):
        m = self._model("CustomModel", model_type=mt)
        assert self._model_can_generate(m) is False

    # ── can_generate() API takes priority ─────────────────────────────────
    def test_can_generate_true_overrides_class_name(self):
        m = self._model("BertForMaskedLM", can_generate_rv=True)
        assert self._model_can_generate(m) is True

    def test_can_generate_false_overrides_class_name(self):
        m = self._model("GPT2ForCausalLM", can_generate_rv=False)
        assert self._model_can_generate(m) is False

    def test_can_generate_exception_falls_back_to_class_name(self):
        cls = type("GPT2LMHeadModel", (object,), {
            "can_generate": lambda self: (_ for _ in ()).throw(RuntimeError("bad"))
        })
        obj = object.__new__(cls)
        obj.config = FakeConfig()
        assert self._model_can_generate(obj) is True


# ---------------------------------------------------------------------------
# 2. TestEncoderDecoderFinalize
# ---------------------------------------------------------------------------

class TestEncoderDecoderFinalize:
    """
    Tests for InternalsRun._finalize() with encoder-decoder step data.
    Data is injected directly (no worker thread) for deterministic shapes.
    """

    def _run(self, input_len=INPUT_LEN):
        from internals_extraction._store import InternalsRun
        return InternalsRun(str(uuid.uuid4()), input_len=input_len)

    def _np(self, *shape):
        return np.random.randn(*shape).astype(np.float32)

    def _enc_step(self, enc_hs, dec_hs, *, has_enc=True):
        return {
            "hidden_states":         dec_hs,
            "encoder_hidden_states": enc_hs if has_enc else None,
            "attentions":            None,
            "encoder_attentions":    None,
            "logits":                None,
        }

    # ── basic structure ────────────────────────────────────────────────────

    def test_encoder_hs_stored(self):
        run = self._run()
        enc = [self._np(BATCH, INPUT_LEN, HIDDEN) for _ in range(NUM_LAYERS + 1)]
        dec = [self._np(BATCH, 1,         HIDDEN) for _ in range(NUM_LAYERS + 1)]
        run._add_step(self._enc_step(enc, dec))
        run._finalize(None, None, None)
        assert run.encoder_hidden_states is not None
        assert len(run.encoder_hidden_states) == NUM_LAYERS + 1
        assert run.encoder_hidden_states[0].shape == (BATCH, INPUT_LEN, HIDDEN)

    def test_input_hs_aliased_to_encoder_hs(self):
        run = self._run()
        enc = [self._np(BATCH, INPUT_LEN, HIDDEN) for _ in range(NUM_LAYERS + 1)]
        dec = [self._np(BATCH, 1,         HIDDEN) for _ in range(NUM_LAYERS + 1)]
        run._add_step(self._enc_step(enc, dec))
        run._finalize(None, None, None)
        assert run.input_hidden_states is run.encoder_hidden_states

    def test_decoder_steps_in_output_hs(self):
        run = self._run()
        enc = [self._np(BATCH, INPUT_LEN, HIDDEN) for _ in range(NUM_LAYERS + 1)]
        dec = [self._np(BATCH, 1,         HIDDEN) for _ in range(NUM_LAYERS + 1)]
        num_steps = 4
        for i in range(num_steps):
            run._add_step(self._enc_step(enc if i == 0 else None, dec, has_enc=(i == 0)))
        run._finalize(None, None, None)
        assert run.output_hidden_states[0].shape == (BATCH, num_steps, HIDDEN)

    def test_num_output_tokens(self):
        run = self._run()
        enc = [self._np(BATCH, INPUT_LEN, HIDDEN) for _ in range(NUM_LAYERS + 1)]
        dec = [self._np(BATCH, 1,         HIDDEN) for _ in range(NUM_LAYERS + 1)]
        for i in range(3):
            run._add_step(self._enc_step(enc if i == 0 else None, dec, has_enc=(i == 0)))
        run._finalize(None, None, None)
        assert run.num_output_tokens == 3

    def test_is_encoder_decoder_true(self):
        run = self._run()
        enc = [self._np(BATCH, INPUT_LEN, HIDDEN) for _ in range(NUM_LAYERS + 1)]
        dec = [self._np(BATCH, 1,         HIDDEN) for _ in range(NUM_LAYERS + 1)]
        run._add_step(self._enc_step(enc, dec))
        run._finalize(None, None, None)
        assert run.is_encoder_decoder is True

    def test_is_encoder_decoder_false_for_decoder_only(self):
        run = self._run()
        hs = [self._np(BATCH, INPUT_LEN, HIDDEN) for _ in range(NUM_LAYERS + 1)]
        run._add_step({
            "hidden_states": hs, "encoder_hidden_states": None,
            "attentions": None, "encoder_attentions": None, "logits": None,
        })
        run._finalize(None, None, None)
        assert run.is_encoder_decoder is False

    # ── means ──────────────────────────────────────────────────────────────

    def test_encoder_hs_mean_shape(self):
        run = self._run()
        enc = [self._np(BATCH, INPUT_LEN, HIDDEN) for _ in range(NUM_LAYERS + 1)]
        dec = [self._np(BATCH, 1,         HIDDEN) for _ in range(NUM_LAYERS + 1)]
        run._add_step(self._enc_step(enc, dec))
        run._finalize(None, None, None)
        assert run.encoder_hidden_states_mean.shape == (NUM_LAYERS + 1, BATCH, HIDDEN)
        assert run.input_hidden_states_mean.shape   == (NUM_LAYERS + 1, BATCH, HIDDEN)

    def test_encoder_hs_mean_values(self):
        run = self._run(input_len=3)
        val = np.full((BATCH, 3, HIDDEN), 4.0, dtype=np.float32)
        enc = [val for _ in range(2)]
        dec = [self._np(BATCH, 1, HIDDEN) for _ in range(2)]
        run._add_step(self._enc_step(enc, dec))
        run._finalize(None, None, None)
        np.testing.assert_allclose(run.encoder_hidden_states_mean[0, 0], 4.0)

    # ── encoder attentions ──────────────────────────────────────────────────

    def test_encoder_attentions_captured(self):
        run = self._run()
        enc    = [self._np(BATCH, INPUT_LEN, HIDDEN) for _ in range(NUM_LAYERS + 1)]
        dec    = [self._np(BATCH, 1,         HIDDEN) for _ in range(NUM_LAYERS + 1)]
        enc_att = [np.random.rand(BATCH, NUM_HEADS, INPUT_LEN, INPUT_LEN)
                   .astype(np.float32) for _ in range(NUM_LAYERS)]
        run._add_step({
            "hidden_states": dec,
            "encoder_hidden_states": enc,
            "attentions": None,
            "encoder_attentions": enc_att,
            "logits": None,
        })
        run._finalize(None, None, None)
        assert run.encoder_attentions is not None
        assert len(run.encoder_attentions) == NUM_LAYERS


# ---------------------------------------------------------------------------
# Common helpers for end-to-end pipeline tests
# ---------------------------------------------------------------------------

def _run_and_wait(model, inputs, fresh_plugin, *, use_generate: bool,
                  max_new_tokens: int = 2, timeout: float = 5.0):
    """
    Run the model (via generate or run_forward), then wait for finalization.
    Returns (run, cfg).
    """
    from internals_extraction._patch import run_forward
    import internals_extraction._patch as _patch_mod
    cfg, store, worker = fresh_plugin

    if use_generate:
        # Monkeypatch _ORIGINAL_GENERATE to our simple loop so we don't need
        # a real transformers generate implementation.
        original = _patch_mod._ORIGINAL_GENERATE
        _patch_mod._ORIGINAL_GENERATE = lambda self, input_ids, **kw: \
            _fake_generate_loop(self, input_ids, max_new_tokens=max_new_tokens, **kw)
        try:
            from internals_extraction._patch import _patched_generate
            _patched_generate(model, inputs["input_ids"])
        finally:
            _patch_mod._ORIGINAL_GENERATE = original
    else:
        run_forward(model, **inputs)

    from internals_extraction._patch import _last_started_run_id
    assert _last_started_run_id is not None
    run = _wait(_last_started_run_id, timeout=timeout)
    return run, cfg


# ---------------------------------------------------------------------------
# 3. TestDecoderOnlyPipeline
# ---------------------------------------------------------------------------

class TestDecoderOnlyPipeline:
    """End-to-end with FakeDecoderOnlyModel (GPT-style)."""

    def _run(self, fresh_plugin, **kw):
        model  = FakeDecoderOnlyModel()
        inputs = _fake_inputs()
        return _run_and_wait(model, inputs, fresh_plugin, use_generate=True,
                             max_new_tokens=2, **kw)

    def test_finalized(self, fresh_plugin):
        run, _ = self._run(fresh_plugin)
        assert run.is_finalized

    def test_is_not_encoder_decoder(self, fresh_plugin):
        run, _ = self._run(fresh_plugin)
        assert run.is_encoder_decoder is False

    def test_input_hidden_states_shape(self, fresh_plugin):
        run, _ = self._run(fresh_plugin)
        assert run.input_hidden_states is not None
        assert len(run.input_hidden_states) == NUM_LAYERS + 1
        assert run.input_hidden_states[0].shape == (BATCH, INPUT_LEN, HIDDEN)

    def test_output_hidden_states_shape(self, fresh_plugin):
        run, _ = self._run(fresh_plugin)
        # 2 generated tokens
        assert run.output_hidden_states[0].shape == (BATCH, 2, HIDDEN)

    def test_means_shape(self, fresh_plugin):
        run, _ = self._run(fresh_plugin)
        assert run.input_hidden_states_mean.shape  == (NUM_LAYERS + 1, BATCH, HIDDEN)
        assert run.output_hidden_states_mean.shape == (NUM_LAYERS + 1, BATCH, HIDDEN)

    def test_logits_shape(self, fresh_plugin):
        # 3 forward calls (step 0 + 2 generated) → 3 sets of logits
        run, _ = self._run(fresh_plugin)
        assert run.logits is not None
        assert run.logits.shape == (3, BATCH, VOCAB)

    def test_encoder_hidden_states_none(self, fresh_plugin):
        run, _ = self._run(fresh_plugin)
        assert run.encoder_hidden_states is None


# ---------------------------------------------------------------------------
# 4. TestEncoderDecoderPipeline
# ---------------------------------------------------------------------------

class TestEncoderDecoderPipeline:
    """End-to-end with FakeEncoderDecoderModel (T5-style)."""

    def _run(self, fresh_plugin, max_new_tokens=2):
        model  = FakeEncoderDecoderModel()
        inputs = _fake_inputs()
        return _run_and_wait(model, inputs, fresh_plugin, use_generate=True,
                             max_new_tokens=max_new_tokens)

    def test_finalized(self, fresh_plugin):
        run, _ = self._run(fresh_plugin)
        assert run.is_finalized

    def test_is_encoder_decoder(self, fresh_plugin):
        run, _ = self._run(fresh_plugin)
        assert run.is_encoder_decoder is True

    def test_encoder_hidden_states_shape(self, fresh_plugin):
        run, _ = self._run(fresh_plugin)
        assert run.encoder_hidden_states is not None
        assert run.encoder_hidden_states[0].shape == (BATCH, INPUT_LEN, HIDDEN)

    def test_input_hs_aliased_to_encoder(self, fresh_plugin):
        run, _ = self._run(fresh_plugin)
        assert run.input_hidden_states is run.encoder_hidden_states

    def test_output_hidden_states_count(self, fresh_plugin):
        """All decoder steps are in output_hidden_states."""
        run, _ = self._run(fresh_plugin, max_new_tokens=3)
        # step 0 + 3 generated = 4 calls, each contributes 1 decoder token
        assert run.output_hidden_states[0].shape[1] == 4

    def test_encoder_hs_mean_shape(self, fresh_plugin):
        run, _ = self._run(fresh_plugin)
        assert run.encoder_hidden_states_mean.shape == (NUM_LAYERS + 1, BATCH, HIDDEN)

    def test_logits_shape(self, fresh_plugin):
        run, _ = self._run(fresh_plugin, max_new_tokens=2)
        assert run.logits is not None
        # 3 forward calls, each produces (batch, 1, VOCAB) → sliced to (batch, VOCAB)
        assert run.logits.shape == (3, BATCH, VOCAB)


# ---------------------------------------------------------------------------
# 5. TestEncoderOnlyPipeline
# ---------------------------------------------------------------------------

class TestEncoderOnlyPipeline:
    """End-to-end with FakeEncoderOnlyModel (BERT base) via run_forward()."""

    def _run(self, fresh_plugin):
        model  = FakeEncoderOnlyModel()
        inputs = _fake_inputs()
        return _run_and_wait(model, inputs, fresh_plugin, use_generate=False)

    def test_finalized(self, fresh_plugin):
        run, _ = self._run(fresh_plugin)
        assert run.is_finalized

    def test_is_not_encoder_decoder(self, fresh_plugin):
        run, _ = self._run(fresh_plugin)
        assert run.is_encoder_decoder is False

    def test_input_hidden_states_shape(self, fresh_plugin):
        run, _ = self._run(fresh_plugin)
        assert run.input_hidden_states is not None
        assert len(run.input_hidden_states) == NUM_LAYERS + 1
        assert run.input_hidden_states[0].shape == (BATCH, INPUT_LEN, HIDDEN)

    def test_no_output_tokens(self, fresh_plugin):
        """Encoder-only has no generation → output_hidden_states is empty."""
        run, _ = self._run(fresh_plugin)
        if run.output_hidden_states is not None:
            assert run.output_hidden_states[0].shape[1] == 0

    def test_no_logits(self, fresh_plugin):
        """Base model has no LM head → logits is None."""
        run, _ = self._run(fresh_plugin)
        assert run.logits is None

    def test_input_hs_mean_shape(self, fresh_plugin):
        run, _ = self._run(fresh_plugin)
        assert run.input_hidden_states_mean.shape == (NUM_LAYERS + 1, BATCH, HIDDEN)


# ---------------------------------------------------------------------------
# 6. TestSequenceClassifierPipeline
# ---------------------------------------------------------------------------

class TestSequenceClassifierPipeline:
    """End-to-end with FakeSequenceClassifier — 2-D logits (batch, num_labels)."""

    def _run(self, fresh_plugin):
        model  = FakeSequenceClassifier()
        inputs = _fake_inputs()
        return _run_and_wait(model, inputs, fresh_plugin, use_generate=False)

    def test_finalized(self, fresh_plugin):
        run, _ = self._run(fresh_plugin)
        assert run.is_finalized

    def test_hidden_states_captured(self, fresh_plugin):
        run, _ = self._run(fresh_plugin)
        assert run.input_hidden_states is not None

    def test_logits_2d_not_crashed(self, fresh_plugin):
        """2-D (batch, num_labels) logits must not crash or be silently dropped."""
        run, _ = self._run(fresh_plugin)
        assert run.logits is not None

    def test_logits_shape(self, fresh_plugin):
        """After stacking 1 step: (1, batch, num_labels)."""
        run, _ = self._run(fresh_plugin)
        assert run.logits.shape == (1, BATCH, FakeSequenceClassifier.NUM_LABELS)

    def test_is_not_encoder_decoder(self, fresh_plugin):
        run, _ = self._run(fresh_plugin)
        assert run.is_encoder_decoder is False


# ---------------------------------------------------------------------------
# 7. TestTokenClassifierPipeline
# ---------------------------------------------------------------------------

class TestTokenClassifierPipeline:
    """End-to-end with FakeTokenClassifier — 3-D logits (batch, seq, labels)."""

    def _run(self, fresh_plugin):
        model  = FakeTokenClassifier()
        inputs = _fake_inputs()
        return _run_and_wait(model, inputs, fresh_plugin, use_generate=False)

    def test_finalized(self, fresh_plugin):
        run, _ = self._run(fresh_plugin)
        assert run.is_finalized

    def test_logits_not_crashed(self, fresh_plugin):
        """3-D non-CausalLM logits must be preserved (not sliced to last token)."""
        run, _ = self._run(fresh_plugin)
        assert run.logits is not None

    def test_logits_shape(self, fresh_plugin):
        """(1 step, batch, seq, num_labels) — full-sequence preserved."""
        run, _ = self._run(fresh_plugin)
        # numpy stack over 1 step of shape (batch, seq, labels)
        # → (1, batch, seq, labels)
        assert run.logits.shape[0] == 1
        assert run.logits.shape[-1] == FakeTokenClassifier.NUM_LABELS

    def test_hidden_states_shape(self, fresh_plugin):
        run, _ = self._run(fresh_plugin)
        assert run.input_hidden_states[0].shape == (BATCH, INPUT_LEN, HIDDEN)


# ---------------------------------------------------------------------------
# 8. TestQAModelPipeline
# ---------------------------------------------------------------------------

class TestQAModelPipeline:
    """End-to-end with FakeQAModel — start_logits / end_logits only."""

    def _run(self, fresh_plugin):
        model  = FakeQAModel()
        inputs = _fake_inputs()
        return _run_and_wait(model, inputs, fresh_plugin, use_generate=False)

    def test_finalized(self, fresh_plugin):
        run, _ = self._run(fresh_plugin)
        assert run.is_finalized

    def test_logits_merged(self, fresh_plugin):
        """start_logits + end_logits must be merged into run.logits."""
        run, _ = self._run(fresh_plugin)
        assert run.logits is not None

    def test_logits_shape(self, fresh_plugin):
        """Merged logits: (1 step, batch=1, 2, seq=INPUT_LEN)."""
        run, _ = self._run(fresh_plugin)
        assert run.logits.shape[0] == 1          # 1 step
        assert run.logits.shape[-2] == 2         # start & end
        assert run.logits.shape[-1] == INPUT_LEN  # seq len

    def test_hidden_states_captured(self, fresh_plugin):
        run, _ = self._run(fresh_plugin)
        assert run.input_hidden_states is not None
        assert len(run.input_hidden_states) == NUM_LAYERS + 1


# ---------------------------------------------------------------------------
# 9. TestMaskedLMPipeline
# ---------------------------------------------------------------------------

class TestMaskedLMPipeline:
    """
    End-to-end with FakeMaskedLMModel (BERT MLM) — 3-D vocab logits kept
    as-is (MaskedLMOutput name → logit_mode='full').
    """

    def _run(self, fresh_plugin):
        model  = FakeMaskedLMModel()
        inputs = _fake_inputs()
        return _run_and_wait(model, inputs, fresh_plugin, use_generate=False)

    def test_finalized(self, fresh_plugin):
        run, _ = self._run(fresh_plugin)
        assert run.is_finalized

    def test_logits_full_sequence_preserved(self, fresh_plugin):
        """MLM logits (batch, seq, vocab) must NOT be sliced to the last token."""
        run, _ = self._run(fresh_plugin)
        assert run.logits is not None
        # (1 step, batch, seq, vocab)
        assert run.logits.shape[-1] == VOCAB
        assert run.logits.shape[-2] == INPUT_LEN  # full sequence, not just last token

    def test_hidden_states_shape(self, fresh_plugin):
        run, _ = self._run(fresh_plugin)
        assert run.input_hidden_states[0].shape == (BATCH, INPUT_LEN, HIDDEN)


# ---------------------------------------------------------------------------
# 10. TestFlagsAcrossModelTypes
# ---------------------------------------------------------------------------

class TestFlagsAcrossModelTypes:
    """Verify extract_* config flags work correctly for each model type."""

    @pytest.mark.parametrize("ModelCls,use_gen", [
        (FakeDecoderOnlyModel, True),
        (FakeEncoderOnlyModel, False),
        (FakeSequenceClassifier, False),
    ])
    def test_hidden_states_off(self, ModelCls, use_gen, fresh_plugin):
        cfg, _, _ = fresh_plugin
        cfg.extract_hidden_states = False

        model  = ModelCls()
        inputs = _fake_inputs()
        run, _ = _run_and_wait(model, inputs, fresh_plugin, use_generate=use_gen)
        assert run.input_hidden_states is None

    @pytest.mark.parametrize("ModelCls,use_gen", [
        (FakeDecoderOnlyModel, True),
        (FakeEncoderOnlyModel, False),
    ])
    def test_logits_off(self, ModelCls, use_gen, fresh_plugin):
        cfg, _, _ = fresh_plugin
        cfg.extract_logits = False

        model  = ModelCls()
        inputs = _fake_inputs()
        run, _ = _run_and_wait(model, inputs, fresh_plugin, use_generate=use_gen)
        assert run.logits is None
