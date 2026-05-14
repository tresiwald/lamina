"""
InternalsStore — thread-safe ring buffer of InternalsRun objects.
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Dict, List, Optional

import numpy as np

from .run import InternalsRun


class InternalsStore:
    """Thread-safe ring buffer of completed InternalsRun objects."""

    def __init__(self, maxlen: int = 10) -> None:
        self._lock = threading.Lock()
        self._runs: deque = deque(maxlen=maxlen)
        self._active: Dict[str, InternalsRun] = {}

    def start_run(
        self,
        run_id: str,
        input_len: int,
        config=None,
        thinking_end_token_id: Optional[int] = None,
    ) -> InternalsRun:
        run = InternalsRun(
            run_id,
            input_len,
            config=config,
            thinking_end_token_id=thinking_end_token_id,
        )
        with self._lock:
            self._active[run_id] = run
        return run

    def add_step(self, run_id: str, step_data: Dict) -> None:
        with self._lock:
            run = self._active.get(run_id)
        if run is not None:
            run._add_step(step_data)

    def end_run(
        self,
        run_id: str,
        lm_head_weight: Optional[np.ndarray] = None,
        lm_head_bias: Optional[np.ndarray] = None,
        final_norm_fn=None,
    ) -> Optional[InternalsRun]:
        with self._lock:
            run = self._active.pop(run_id, None)
        if run is not None:
            run._finalize(lm_head_weight, lm_head_bias, final_norm_fn)
            with self._lock:
                self._runs.append(run)
        return run

    def get_latest(self) -> Optional[InternalsRun]:
        with self._lock:
            return self._runs[-1] if self._runs else None

    def get_run(self, run_id: str) -> Optional[InternalsRun]:
        with self._lock:
            for run in reversed(self._runs):
                if run.run_id == run_id:
                    return run
        return None

    def get_all(self) -> List[InternalsRun]:
        with self._lock:
            return list(self._runs)

    def resize(self, maxlen: int) -> None:
        with self._lock:
            self._runs = deque(self._runs, maxlen=maxlen)

    def __len__(self) -> int:
        with self._lock:
            return len(self._runs)
