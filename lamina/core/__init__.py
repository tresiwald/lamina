"""
lamina.core
===========
Library-agnostic building blocks: configuration, run data-container,
ring-buffer store, and background worker thread.

Nothing in this sub-package imports from any ML framework.
"""
from .config import InternalsConfig
from .run import InternalsRun
from .store import InternalsStore
from .worker import BackgroundWorker

__all__ = [
    "InternalsConfig",
    "InternalsRun",
    "InternalsStore",
    "BackgroundWorker",
]
