"""
Configuration dataclass for lamina.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InternalsConfig:
    """
    Controls what gets extracted and how it is stored.

    All fields have sensible defaults; mutate fields before running
    inference to change behaviour at runtime::

        import lamina
        lamina.config.extract_logit_lens = True

    Extraction flags
    ----------------
    extract_hidden_states : bool
        Capture the full ``(num_layers + 1, batch, seq, hidden_dim)``
        hidden-state stack at every generate step.  Layer 0 is the
        embedding, layers 1..N are transformer block outputs.
    extract_attentions : bool
        Capture per-layer attention weight matrices.
    extract_logits : bool
        Capture the LM-head output (logits) for the last token at
        every generate step.
    extract_logit_lens : bool
        Apply the model's final LayerNorm + LM-head to every layer's
        hidden state to get "logit-lens" predictions.  Done entirely on
        CPU in the background thread.  Requires
        ``lamina.register_model(model)`` after loading.

    Aggregation flags
    -----------------
    aggregate_attention_heads : bool
        Average attention weights across the head dimension.
        True  → stored shape is ``(batch, seq_q, seq_k)`` per layer.
        False → stored shape is ``(batch, heads, seq_q, seq_k)``.

    Storage
    -------
    max_stored_runs : int
        Number of completed inference runs kept in the ring buffer.
        Older runs are evicted automatically.

    Worker
    ------
    worker_queue_maxsize : int
        Maximum number of pending items in the processing queue.
        0 means unbounded (default).
    """

    # ── extraction ───────────────────────────────────────────────────────────
    extract_hidden_states: bool = True
    extract_attentions: bool = True
    extract_logits: bool = True
    extract_logit_lens: bool = False   # off by default — needs register_model()

    # ── aggregation ──────────────────────────────────────────────────────────
    aggregate_attention_heads: bool = True

    # ── storage ──────────────────────────────────────────────────────────────
    max_stored_runs: int = 10

    # ── worker ───────────────────────────────────────────────────────────────
    worker_queue_maxsize: int = 0

    def __repr__(self) -> str:  # pragma: no cover
        flags = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"InternalsConfig({flags})"
