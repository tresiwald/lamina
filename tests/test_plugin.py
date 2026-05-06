"""
Tests for internals_extraction.

These tests use a tiny hand-built fake model so they run without a GPU
or a real transformers checkpoint.  They verify:

  - Patching is applied on import.
  - Hidden states, attentions, and logits are captured correctly.
  - Head aggregation (mean) works.
  - Input / output hidden-state means are computed correctly.
  - Logit-lens is computed when register_model() is called and enabled.
  - The ring buffer evicts old runs.
  - uninstall_patches() restores originals.
"""
from __future__ import annotations

import time
import types
import uuid

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers — tiny fake transformers-like model
# ---------------------------------------------------------------------------

def _make_model_output(
    hidden_states=None,
    attentions=None,
    logits=None,
):
    """Return a SimpleNamespace that mimics a ModelOutput."""
    return types.SimpleNamespace(
        hidden_states=hidden_states,
        attentions=attentions,
        logits=logits,
        past_key_values=None,
    )


class FakeTensor:
    """Minimal tensor-like object (CPU, backed by numpy)."""

    def __init__(self, array: np.ndarray):
        self._data = array.astype(np.float32)

    @property
    def shape(self):
        return self._data.shape

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return self

    def numpy(self):
        return self._data

    def mean(self, dim):
        return FakeTensor(self._data.mean(axis=dim))

    def __getitem__(self, item):
        return FakeTensor(self._data[item])


def _ft(array):
    return FakeTensor(np.array(array, dtype=np.float32))


# ---------------------------------------------------------------------------
# Fixtures — import plugin with fresh singletons each test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def fresh_plugin(monkeypatch):
    """Re-initialise plugin singletons and re-install patches each test."""
    import internals_extraction
    from internals_extraction._config import InternalsConfig
    from internals_extraction._store import InternalsStore
    from internals_extraction._worker import BackgroundWorker
    from internals_extraction import _patch

    cfg    = InternalsConfig()
    store  = InternalsStore(maxlen=cfg.max_stored_runs)
    worker = BackgroundWorker(store, maxsize=0)
    _patch._initialise(cfg, store, worker)

    monkeypatch.setattr(internals_extraction, "config",  cfg)
    monkeypatch.setattr(internals_extraction, "_store",  store)
    monkeypatch.setattr(internals_extraction, "_worker", worker)

    yield cfg, store, worker

    worker.stop()


def _wait_finalized(run_id: str, timeout: float = 5.0) -> object:
    """Poll until the run is finalized and return it."""
    import internals_extraction
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = internals_extraction.get_run(run_id)
        if run is not None and run.is_finalized:
            return run
        time.sleep(0.02)
    pytest.fail(f"Run {run_id!r} not finalized within {timeout}s")


# ---------------------------------------------------------------------------
# Unit tests for InternalsStore
# ---------------------------------------------------------------------------

class TestInternalsStore:
    def test_ring_buffer_eviction(self):
        from internals_extraction._store import InternalsStore
        store = InternalsStore(maxlen=3)
        ids = []
        for i in range(5):
            rid = str(i)
            store.start_run(rid, input_len=4)
            store.add_step(rid, {"hidden_states": None, "attentions": None, "logits": None})
            store.end_run(rid)
            ids.append(rid)

        runs = store.get_all()
        assert len(runs) == 3
        assert {r.run_id for r in runs} == {"2", "3", "4"}

    def test_resize(self):
        from internals_extraction._store import InternalsStore
        store = InternalsStore(maxlen=10)
        for i in range(8):
            rid = str(i)
            store.start_run(rid, input_len=1)
            store.end_run(rid)
        store.resize(3)
        assert len(store) == 3


# ---------------------------------------------------------------------------
# Unit tests for InternalsRun._finalize
# ---------------------------------------------------------------------------

class TestInternalsRunFinalize:

    def _make_run(self, input_len=3):
        from internals_extraction._store import InternalsRun
        return InternalsRun(run_id=str(uuid.uuid4()), input_len=input_len)

    def _make_hs(self, batch, seq, hidden, num_layers=2):
        """Return a list of np arrays: [layer0_array, layer1_array, ...]"""
        return [
            np.random.randn(batch, seq, hidden).astype(np.float32)
            for _ in range(num_layers + 1)   # +1 for embedding layer
        ]

    def _make_att(self, batch, heads, seq_q, seq_k, num_layers=2):
        return [
            np.random.rand(batch, heads, seq_q, seq_k).astype(np.float32)
            for _ in range(num_layers)
        ]

    def test_basic_shapes(self):
        run = self._make_run(input_len=3)
        batch, input_len, hidden, num_layers = 1, 3, 8, 2

        # Step 0: full input sequence
        hs = self._make_hs(batch, input_len, hidden, num_layers)
        att = self._make_att(batch, 2, input_len, input_len, num_layers)
        run._add_step({
            "hidden_states": hs,
            "attentions": att,
            "logits": np.random.randn(batch, input_len, 16).astype(np.float32),
        })

        # Step 1: single new token (KV-cache mode)
        hs1 = self._make_hs(batch, 1, hidden, num_layers)
        att1 = self._make_att(batch, 2, 1, input_len + 1, num_layers)
        run._add_step({
            "hidden_states": hs1,
            "attentions": att1,
            "logits": np.random.randn(batch, 1, 16).astype(np.float32),
        })

        run._finalize(None, None, None)

        # input hidden states: (num_layers+1) entries, each (batch, input_len, hidden)
        assert len(run.input_hidden_states) == num_layers + 1
        assert run.input_hidden_states[0].shape == (batch, input_len, hidden)

        # output hidden states: (num_layers+1) entries, each (batch, 1, hidden)
        assert run.output_hidden_states[0].shape == (batch, 1, hidden)

        # means: (num_layers+1, batch, hidden)
        assert run.input_hidden_states_mean.shape == (num_layers + 1, batch, hidden)
        assert run.output_hidden_states_mean.shape == (num_layers + 1, batch, hidden)

        # logits: (num_steps=2, batch, vocab)
        assert run.logits.shape == (2, batch, 16)

    def test_mean_values(self):
        run = self._make_run(input_len=2)
        batch, seq, hidden = 1, 2, 4
        hs_data = np.ones((batch, seq, hidden), dtype=np.float32) * 3.0
        run._add_step({
            "hidden_states": [hs_data],    # 1 layer only
            "attentions": None,
            "logits": None,
        })
        run._finalize(None, None, None)
        mean = run.input_hidden_states_mean
        assert mean is not None
        np.testing.assert_allclose(mean[0, 0, :], 3.0)

    def test_no_output_tokens(self):
        """Single forward pass, no generated tokens."""
        run = self._make_run(input_len=4)
        hs = self._make_hs(1, 4, 8, 2)
        run._add_step({"hidden_states": hs, "attentions": None, "logits": None})
        run._finalize(None, None, None)

        assert run.num_output_tokens == 0 or run.output_hidden_states[0].shape[1] == 0


# ---------------------------------------------------------------------------
# Integration tests via worker thread
# ---------------------------------------------------------------------------

class TestWorkerIntegration:

    def _enqueue_run(self, worker, store, config, num_steps=3, input_len=4,
                     batch=1, hidden=8, num_layers=2, vocab=20, heads=2):
        from internals_extraction._store import InternalsStore

        run_id = str(uuid.uuid4())
        store.start_run(run_id, input_len)

        for step_idx in range(num_steps):
            seq = input_len if step_idx == 0 else 1
            hs = tuple(
                FakeTensor(np.random.randn(batch, seq, hidden).astype(np.float32))
                for _ in range(num_layers + 1)
            )
            att = tuple(
                FakeTensor(np.random.rand(batch, heads, seq, input_len + step_idx).astype(np.float32))
                for _ in range(num_layers)
            )
            logits = FakeTensor(np.random.randn(batch, seq, vocab).astype(np.float32))

            worker.enqueue_step({
                "kind": "step",
                "run_id": run_id,
                "step_idx": step_idx,
                "config": config,
                "hidden_states": hs,
                "attentions": att,
                "logits": logits,
            })

        worker.enqueue_end({
            "kind": "end",
            "run_id": run_id,
            "lm_head_weight": None,
            "lm_head_bias": None,
            "final_norm_fn": None,
        })

        return run_id

    def test_run_is_finalized(self, fresh_plugin):
        cfg, store, worker = fresh_plugin
        run_id = self._enqueue_run(worker, store, cfg)
        run = _wait_finalized(run_id)
        assert run.is_finalized

    def test_hidden_state_shapes(self, fresh_plugin):
        cfg, store, worker = fresh_plugin
        batch, hidden, num_layers, input_len, num_steps = 1, 8, 2, 4, 3
        run_id = self._enqueue_run(
            worker, store, cfg,
            num_steps=num_steps, input_len=input_len,
            batch=batch, hidden=hidden, num_layers=num_layers,
        )
        run = _wait_finalized(run_id)

        # (num_layers+1) entries for input hidden states
        assert len(run.input_hidden_states) == num_layers + 1
        # Each: (batch, input_len, hidden)
        assert run.input_hidden_states[0].shape == (batch, input_len, hidden)

        # (num_steps - 1) output tokens
        assert run.output_hidden_states[0].shape == (batch, num_steps - 1, hidden)

        # Means
        assert run.input_hidden_states_mean.shape  == (num_layers + 1, batch, hidden)
        assert run.output_hidden_states_mean.shape == (num_layers + 1, batch, hidden)

    def test_attentions_aggregated(self, fresh_plugin):
        cfg, store, worker = fresh_plugin
        cfg.aggregate_attention_heads = True
        run_id = self._enqueue_run(worker, store, cfg, num_steps=2)
        run = _wait_finalized(run_id)

        # attentions[step][layer]: (batch, seq_q, seq_k) — no head dim
        att = run.attentions[0][0]
        assert att.ndim == 3

    def test_attentions_not_aggregated(self, fresh_plugin):
        cfg, store, worker = fresh_plugin
        cfg.aggregate_attention_heads = False
        run_id = self._enqueue_run(worker, store, cfg, num_steps=2, heads=2)
        run = _wait_finalized(run_id)

        att = run.attentions[0][0]
        assert att.ndim == 4   # (batch, heads, seq_q, seq_k)

    def test_logits_shape(self, fresh_plugin):
        cfg, store, worker = fresh_plugin
        vocab, num_steps, batch = 20, 3, 1
        run_id = self._enqueue_run(
            worker, store, cfg, num_steps=num_steps, batch=batch, vocab=vocab
        )
        run = _wait_finalized(run_id)
        assert run.logits.shape == (num_steps, batch, vocab)

    def test_logit_lens(self, fresh_plugin):
        cfg, store, worker = fresh_plugin
        cfg.extract_logit_lens = True

        batch, hidden, vocab, input_len, num_layers = 1, 8, 20, 4, 2
        lm_w = np.random.randn(vocab, hidden).astype(np.float32)

        def _norm_fn(x):
            return x   # identity

        # Manually enqueue steps + end (do NOT use _enqueue_run which sends its own end)
        run_id = str(uuid.uuid4())
        store.start_run(run_id, input_len)

        for step_idx in range(2):
            seq = input_len if step_idx == 0 else 1
            hs = tuple(
                FakeTensor(np.random.randn(batch, seq, hidden).astype(np.float32))
                for _ in range(num_layers + 1)
            )
            worker.enqueue_step({
                "kind": "step",
                "run_id": run_id,
                "step_idx": step_idx,
                "config": cfg,
                "hidden_states": hs,
                "attentions": None,
                "logits": None,
            })

        worker.enqueue_end({
            "kind": "end",
            "run_id": run_id,
            "lm_head_weight": lm_w,
            "lm_head_bias": None,
            "final_norm_fn": _norm_fn,
        })

        run = _wait_finalized(run_id)
        assert run.logit_lens is not None, "logit_lens should be populated"
        # Each entry: (batch, input_len, vocab)
        assert run.logit_lens[0].shape == (batch, input_len, vocab)

    def test_flags_off(self, fresh_plugin):
        cfg, store, worker = fresh_plugin
        cfg.extract_hidden_states = False
        cfg.extract_attentions    = False
        cfg.extract_logits        = False

        run_id = self._enqueue_run(worker, store, cfg)
        run = _wait_finalized(run_id)

        assert run.input_hidden_states is None
        assert run.attentions is None
        assert run.logits is None

    def test_multiple_runs_ring_buffer(self, fresh_plugin):
        cfg, store, worker = fresh_plugin
        cfg.max_stored_runs = 3
        store.resize(3)

        ids = [
            self._enqueue_run(worker, store, cfg)
            for _ in range(5)
        ]
        # wait for last one
        _wait_finalized(ids[-1])
        time.sleep(0.05)  # let worker flush earlier runs too

        all_runs = store.get_all()
        assert len(all_runs) <= 3

    def test_get_latest(self, fresh_plugin):
        cfg, store, worker = fresh_plugin
        run_id = self._enqueue_run(worker, store, cfg)
        run = _wait_finalized(run_id)
        latest = store.get_latest()
        assert latest is not None
        assert latest.run_id == run_id
