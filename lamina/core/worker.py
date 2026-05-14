"""
BackgroundWorker — daemon thread that converts GPU tensors to NumPy
and forwards processed step data to InternalsStore.

Queue item protocol
-------------------
Each item is a dict with one of three shapes:

  { "kind": "step",
    "run_id": str,
    "step_idx": int,
    "hidden_states": tuple[Tensor] | None,
    "encoder_hidden_states": tuple[Tensor] | None,
    "attentions": tuple[Tensor] | None,
    "encoder_attentions": tuple[Tensor] | None,
    "logits": Tensor | None,
    "start_logits": Tensor | None,     # QA models
    "end_logits": Tensor | None,       # QA models
    "logit_mode": str,                 # "last_token" | "full"
    "config": InternalsConfig }

  { "kind": "end",
    "run_id": str,
    "lm_head_weight": np.ndarray | None,
    "lm_head_bias": np.ndarray | None,
    "final_norm_fn": callable | None }

  { "kind": "poison" }   ← shuts the thread down (used in tests)
"""
from __future__ import annotations

import queue
import threading
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .store import InternalsStore
    from .config import InternalsConfig


def _to_numpy(tensor) -> np.ndarray:
    """Detach, move to CPU, convert to NumPy — safe on any device/dtype.

    NumPy has no bfloat16 type, so bfloat16 tensors are cast to float32
    before conversion.  All other dtypes are passed through unchanged.
    """
    import torch
    t = tensor.detach().cpu()
    if t.dtype == torch.bfloat16:
        t = t.float()
    return t.numpy()


def _process_step(item: dict, store: "InternalsStore") -> None:
    config: "InternalsConfig" = item["config"]
    run_id: str = item["run_id"]
    step_data: dict = {}

    # ── Hidden states ─────────────────────────────────────────────────────────
    if config.extract_hidden_states and item.get("hidden_states") is not None:
        step_data["hidden_states"] = [
            _to_numpy(hs) for hs in item["hidden_states"]
        ]
    else:
        step_data["hidden_states"] = None

    # ── Encoder hidden states (encoder-decoder, step 0 only) ──────────────────
    if config.extract_hidden_states and item.get("encoder_hidden_states") is not None:
        step_data["encoder_hidden_states"] = [
            _to_numpy(hs) for hs in item["encoder_hidden_states"]
        ]
    else:
        step_data["encoder_hidden_states"] = None

    # ── Attentions ────────────────────────────────────────────────────────────
    if config.extract_attentions and item.get("attentions") is not None:
        attn_list = []
        for attn in item["attentions"]:
            arr = _to_numpy(attn)
            if config.aggregate_attention_heads:
                arr = arr.mean(axis=1)
            attn_list.append(arr)
        step_data["attentions"] = attn_list
    else:
        step_data["attentions"] = None

    # ── Encoder attentions ────────────────────────────────────────────────────
    if config.extract_attentions and item.get("encoder_attentions") is not None:
        enc_attn_list = []
        for attn in item["encoder_attentions"]:
            arr = _to_numpy(attn)
            if config.aggregate_attention_heads:
                arr = arr.mean(axis=1)
            enc_attn_list.append(arr)
        step_data["encoder_attentions"] = enc_attn_list
    else:
        step_data["encoder_attentions"] = None

    # ── Logits ───────────────────────────────────────────────────────────────
    if config.extract_logits:
        logits_tensor = item.get("logits")
        start_tensor  = item.get("start_logits")
        end_tensor    = item.get("end_logits")

        if logits_tensor is not None:
            logits_arr = _to_numpy(logits_tensor)
            mode = item.get("logit_mode", "last_token")
            if mode == "last_token" and logits_arr.ndim == 3:
                step_data["logits"] = logits_arr[:, -1, :]
            else:
                step_data["logits"] = logits_arr
            step_data["logit_mode"] = mode
        elif start_tensor is not None and end_tensor is not None:
            # QA: merge start/end → (batch, 2, seq)
            start_arr = _to_numpy(start_tensor)
            end_arr   = _to_numpy(end_tensor)
            step_data["logits"] = np.stack([start_arr, end_arr], axis=1)
            step_data["logit_mode"] = "full"
        else:
            step_data["logits"] = None
            step_data["logit_mode"] = "last_token"
    else:
        step_data["logits"] = None
        step_data["logit_mode"] = "last_token"

    store.add_step(run_id, step_data)


def _process_end(item: dict, store: "InternalsStore") -> None:
    store.end_run(
        run_id=item["run_id"],
        lm_head_weight=item.get("lm_head_weight"),
        lm_head_bias=item.get("lm_head_bias"),
        final_norm_fn=item.get("final_norm_fn"),
    )


def _worker_loop(q: queue.Queue, store: "InternalsStore") -> None:
    """Main loop executed in the background daemon thread."""
    while True:
        try:
            item = q.get(timeout=1.0)
        except queue.Empty:
            continue

        try:
            kind = item.get("kind")
            if kind == "step":
                _process_step(item, store)
            elif kind == "end":
                _process_end(item, store)
            elif kind == "poison":
                break
        except Exception as exc:  # noqa: BLE001
            import sys
            print(
                f"[lamina] worker error on {item.get('kind')!r}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
        finally:
            q.task_done()


class BackgroundWorker:
    """Owns the queue and the daemon thread."""

    def __init__(self, store: "InternalsStore", maxsize: int = 0) -> None:
        self._store = store
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._thread = threading.Thread(
            target=_worker_loop,
            args=(self._queue, self._store),
            name="lamina-worker",
            daemon=True,
        )
        self._thread.start()

    def enqueue_step(self, item: dict) -> None:
        """Non-blocking put; drops the item if the queue is full."""
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            import sys
            print(
                "[lamina] WARNING: processing queue full — step data dropped. "
                "Increase worker_queue_maxsize.",
                file=sys.stderr,
            )

    def enqueue_end(self, item: dict) -> None:
        """Blocking put to ensure finalisation is always enqueued."""
        self._queue.put(item)

    def stop(self) -> None:
        """Signal the worker to stop cleanly (used in tests)."""
        self._queue.put({"kind": "poison"})
        self._thread.join(timeout=5.0)

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()
