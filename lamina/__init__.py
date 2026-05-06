"""
lamina
======
A modular library for extracting and examining the internal representations
of neural language models — without modifying model code and without slowing
down GPU inference.

    "It is the little grey cells, mon ami." — Hercule Poirot

Architecture
------------
::

    lamina.core          Library-agnostic: InternalsRun, InternalsStore,
                         BackgroundWorker, InternalsConfig
    lamina.extractors    Framework-specific capture:
      .hf                HuggingFace Transformers (GenerationMixin hooks)
      .vllm              vLLM (planned)
    lamina.interventions Active model modifications (planned):
                         ActivationPatch, AttentionMask, SteeringVector
    lamina.backends      Pluggable storage:
      .filesystem        .npz + metadata.jsonl (default)
      .hf_dataset        HuggingFace datasets.Dataset
    lamina.applications  Higher-level APIs:
      .dataset           InternalsDataset / InternalsInstance / InternalsRecord
      .spans             TextSpan, SpanSpec, offset-mapping resolver

How it works
------------
1. On import, a background daemon thread starts and the HF extractor
   installs a monkey-patch on ``GenerationMixin.generate``.
2. Every ``model.generate()`` call registers a ``register_forward_hook``
   on the model instance; the hook captures output tensors (still on GPU)
   and enqueues them without blocking inference.
3. The daemon thread converts tensors to CPU NumPy arrays and finalises
   each ``InternalsRun`` in the ring buffer.

Quick start — single inference
--------------------------------
::

    import lamina
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model     = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    inputs = tokenizer("Hello, world", return_tensors="pt")
    model.generate(**inputs, max_new_tokens=20)

    run = lamina.get_latest()
    print(run.input_hidden_states_mean.shape)   # (13, 1, 768)

Quick start — dataset processing
----------------------------------
::

    from lamina import InternalsDataset, InternalsInstance
    from lamina.backends import dump, load

    dataset = InternalsDataset([
        InternalsInstance(text="The cat sat.", properties={"label": 1}),
        InternalsInstance(text="Dogs bark.",   properties={"label": 0}),
    ])

    records = dataset.run(model, tokenizer, generate_kwargs={"max_new_tokens": 1})
    dump(records, "output/")

    arrays_list, metadata_list = load("output/")

Quick start — encoder-only model
-----------------------------------
::

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from lamina import run_forward

    model     = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    inputs = tokenizer("Hello world", return_tensors="pt")
    run_forward(model, **inputs)

    run = lamina.get_latest()

Public API
----------
.. code-block:: text

    # Inference capture
    lamina.config                InternalsConfig  (mutable)
    lamina.get_latest()          → InternalsRun | None
    lamina.get_run(run_id)       → InternalsRun | None
    lamina.get_all()             → list[InternalsRun]
    lamina.wait_for_run(id)      → InternalsRun | None
    lamina.set_config(**kw)      → None
    lamina.run_forward(model, …) → ModelOutput   (non-generative models)
    lamina.register_model(m)     → None          (logit-lens)

    # Dataset processing
    InternalsInstance(text, properties, spans)
    InternalsDataset(instances).run(model, tokenizer, …)
    InternalsRecord.properties / .spans / .run / .span_hidden_states_mean

    # Serialisation
    from lamina.backends import dump, load, to_hf_dataset
    dump(records, outdir)
    load(outdir) → (arrays_list, metadata_list)

    # Interventions (planned)
    from lamina.interventions import Intervention
"""
from __future__ import annotations

from typing import List, Optional

# ── Core ──────────────────────────────────────────────────────────────────────
from .core.config import InternalsConfig
from .core.run    import InternalsRun
from .core.store  import InternalsStore
from .core.worker import BackgroundWorker

# ── Extractors ────────────────────────────────────────────────────────────────
from .extractors.hf.extractor   import (
    _initialise,
    install_patches,
    uninstall_patches,
    run_forward,
)
from .extractors.hf.lm_head     import register_model
from .extractors.hf.model_detect import _model_can_generate

# ── Interventions ─────────────────────────────────────────────────────────────
from .interventions.base import Intervention

# ── Backends ──────────────────────────────────────────────────────────────────
from .backends.filesystem  import dump, load
from .backends.hf_dataset  import to_hf_dataset

# ── Applications ──────────────────────────────────────────────────────────────
from .applications.spans   import SpanSpec, TextSpan, SpanResolutionError
from .applications.dataset import InternalsInstance, InternalsRecord, InternalsDataset

__all__ = [
    # Core types
    "InternalsConfig",
    "InternalsRun",
    # Inference capture
    "config",
    "get_latest",
    "get_run",
    "get_all",
    "wait_for_run",
    "set_config",
    "run_forward",
    "register_model",
    # Interventions
    "Intervention",
    # Dataset API
    "SpanSpec",
    "TextSpan",
    "SpanResolutionError",
    "InternalsInstance",
    "InternalsRecord",
    "InternalsDataset",
    # Serialisation
    "dump",
    "load",
    "to_hf_dataset",
]

# ---------------------------------------------------------------------------
# Singleton setup — runs exactly once on first import
# ---------------------------------------------------------------------------

#: Global configuration.  Mutate fields before running inference.
config: InternalsConfig = InternalsConfig()

_store:  InternalsStore   = InternalsStore(maxlen=config.max_stored_runs)
_worker: BackgroundWorker = BackgroundWorker(_store, maxsize=config.worker_queue_maxsize)

_initialise(config, _store, _worker)
install_patches()


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_latest() -> Optional[InternalsRun]:
    """Return the most recently completed run, or ``None``."""
    return _store.get_latest()


def get_run(run_id: str) -> Optional[InternalsRun]:
    """Return the run with the given UUID, or ``None``."""
    return _store.get_run(run_id)


def get_all() -> List[InternalsRun]:
    """Return all stored runs, oldest first."""
    return _store.get_all()


def wait_for_run(run_id: str, timeout: float = 30.0) -> Optional[InternalsRun]:
    """Block until the run is finalised; return ``None`` on timeout."""
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = _store.get_run(run_id)
        if run is not None and run.is_finalized:
            return run
        time.sleep(0.01)
    return None


def set_config(**kwargs) -> None:
    """
    Update multiple config fields at once::

        lamina.set_config(
            extract_logit_lens=True,
            aggregate_attention_heads=False,
            max_stored_runs=50,
        )
    """
    for key, value in kwargs.items():
        if not hasattr(config, key):
            raise AttributeError(f"InternalsConfig has no field {key!r}")
        setattr(config, key, value)
    if "max_stored_runs" in kwargs:
        _store.resize(kwargs["max_stored_runs"])
