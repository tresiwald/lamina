```
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║    ██╗███╗   ██╗████████╗███████╗██████╗ ███╗   ██╗ █████╗ ██╗         ║
║    ██║████╗  ██║╚══██╔══╝██╔════╝██╔══██╗████╗  ██║██╔══██╗██║         ║
║    ██║██╔██╗ ██║   ██║   █████╗  ██████╔╝██╔██╗ ██║███████║██║         ║
║    ██║██║╚██╗██║   ██║   ██╔══╝  ██╔══██╗██║╚██╗██║██╔══██║██║         ║
║    ██║██║ ╚████║   ██║   ███████╗██║  ██║██║ ╚████║██║  ██║███████╗    ║
║    ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝   ║
║                                                                          ║
║    ███████╗██╗  ██╗████████╗██████╗  █████╗  ██████╗████████╗██╗ ██████╗███╗   ██╗
║    ██╔════╝╚██╗██╔╝╚══██╔══╝██╔══██╗██╔══██╗██╔════╝╚══██╔══╝██║██╔═══██╗████╗  ██║
║    █████╗   ╚███╔╝    ██║   ██████╔╝███████║██║        ██║   ██║██║   ██║██╔██╗ ██║
║    ██╔══╝   ██╔██╗    ██║   ██╔══██╗██╔══██║██║        ██║   ██║██║   ██║██║╚██╗██║
║    ███████╗██╔╝ ██╗   ██║   ██║  ██║██║  ██║╚██████╗   ██║   ██║╚██████╔╝██║ ╚████║
║    ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
║                                                                          ║
║   Drop-in plugin for HuggingFace Transformers.                          ║
║   Layer-wise hidden states · attentions · logits · logit-lens           ║
║   Zero model changes · zero inference slowdown · zero GPU overhead      ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## What it extracts

| Signal | Shape | Description |
|---|---|---|
| `input_hidden_states` | `list[layer] → (batch, input_len, hidden)` | Full hidden states for every prompt token, per layer |
| `output_hidden_states` | `list[layer] → (batch, output_tokens, hidden)` | Full hidden states for every generated token, per layer |
| `input_hidden_states_mean` | `(num_layers, batch, hidden)` | Per-layer mean over the prompt sequence |
| `output_hidden_states_mean` | `(num_layers, batch, hidden)` | Per-layer mean over generated tokens |
| `attentions` | `list[step][layer] → (batch, [heads,] seq_q, seq_k)` | Per-layer attention weights, optionally head-aggregated |
| `logits` | `(steps, batch, vocab_size)` | LM-head output at the last position of each step |
| `logit_lens` | `list[layer] → (batch, input_len, vocab_size)` | Logit-lens: LM-head applied to each layer's hidden state *(opt-in)* |
| `span_<name>` | `(num_layers, hidden)` | Per-layer mean hidden state averaged over a named token span *(opt-in)* |

Layer index `0` is always the embedding output; layers `1..N` are transformer block outputs.

---

## How it works

```
  model.generate()
       │
       ▼
  ┌─────────────────────────────────────────────────────────┐
  │  patched generate  (installed on import, no user code)  │
  │                                                         │
  │  ① forces output_hidden_states / output_attentions      │
  │  ② registers a forward hook on the model instance       │
  │  ③ calls original generate  ← GPU runs at full speed    │
  │                                                         │
  │  per-step forward hook:                                 │
  │    • grabs tensor references  ← nanoseconds             │
  │    • enqueues them            ← non-blocking            │
  └─────────────────────────────────────────────────────────┘
       │  queue (never blocks the GPU)
       ▼
  ┌─────────────────────────────────────────────────────────┐
  │  background daemon thread                               │
  │                                                         │
  │  • tensor.detach().cpu().numpy()  ← GPU freed per step  │
  │  • aggregates hidden states / attentions / logits       │
  │  • finalizes InternalsRun into thread-safe ring buffer  │
  └─────────────────────────────────────────────────────────┘
       │
       ▼
  internals_extraction.get_latest()   ← your code
```

**GPU memory:** hidden-state tensors are materialised on GPU during the forward pass (unavoidable), then immediately moved to CPU in the background thread. Peak overhead = one generation step's worth of hidden states, not the full history.

---

## Installation

```bash
pip install -e .
# with HuggingFace datasets support:
pip install -e ".[datasets]"
```

Requirements: `numpy >= 1.21`, `transformers >= 4.30`.

---

## Notebooks

| Notebook | Topics |
|---|---|
| [`notebooks/01_quickstart.ipynb`](notebooks/01_quickstart.ipynb) | Import, run, inspect all signals |
| [`notebooks/02_spans.ipynb`](notebooks/02_spans.ipynb) | TextSpan, SpanSpec, occurrence, list form |
| [`notebooks/03_dataset_processing.ipynb`](notebooks/03_dataset_processing.ipynb) | InternalsDataset, dump/load, properties |
| [`notebooks/04_hf_datasets.ipynb`](notebooks/04_hf_datasets.ipynb) | from_hf_dataset, to_hf_dataset, push_to_hub |

---

## Quick start

```python
import internals_extraction                           # ← this is all you need
from transformers import AutoModelForCausalLM, AutoTokenizer

model     = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

inputs = tokenizer("The Eiffel Tower is located in", return_tensors="pt")
model.generate(**inputs, max_new_tokens=20)

run = internals_extraction.get_latest()
print(run)
# InternalsRun(run_id='...', input_len=7, num_output_tokens=20,
#              num_layers=13, finalized=True)

# Layer-wise mean hidden state of prompt tokens — (13, 1, 768)
print(run.input_hidden_states_mean.shape)

# Layer-wise mean hidden state of generated tokens — (13, 1, 768)
print(run.output_hidden_states_mean.shape)

# Full per-layer hidden states for prompt — list of 13 arrays, each (1, 7, 768)
print(run.input_hidden_states[6].shape)

# Attention weights at step 0, layer 0 — (1, seq_q, seq_k)
print(run.attentions[0][0].shape)

# Logits at every generation step — (20, 1, 50257)
print(run.logits.shape)
```

---

## Dataset processing

The dataset layer lets you process a list of items, attach task-specific metadata, define named token spans, and dump everything to disk (or a HuggingFace `Dataset`).

### InternalsInstance

```python
from internals_extraction import InternalsInstance, TextSpan, SpanSpec

instance = InternalsInstance(
    text="The cat sat on the mat.",
    properties={"label": 1, "id": "ex-001", "task": "nli"},
    spans={
        "subject":   "The cat",          # substring → resolved via offset mapping
        "predicate": "sat on the mat",   # substring
    },
)
```

### InternalsDataset

```python
from internals_extraction import InternalsDataset, InternalsInstance

dataset = InternalsDataset([
    InternalsInstance(
        text="Paris is the capital of France.",
        properties={"label": 1, "id": 0},
        spans={"entity": "Paris", "country": "France"},
    ),
    InternalsInstance(
        text="The sky is blue.",
        properties={"label": 0, "id": 1},
    ),
])

records = dataset.run(model, tokenizer, generate_kwargs={"max_new_tokens": 1})

for rec in records:
    print(rec.properties)                              # {"label": ..., "id": ...}
    print(rec.run.input_hidden_states_mean.shape)      # (num_layers, 1, hidden)
    if rec.span_hidden_states_mean:
        print(rec.span_hidden_states_mean["entity"])   # (num_layers, hidden)
    print(rec.resolved_spans)                          # {"entity": SpanSpec(0,1), ...}
```

### Dump and load

```python
from internals_extraction import dump, load

dump(records, "output/my_run/")

# Output layout:
# output/my_run/
#   metadata.jsonl       one JSON line per record
#   00000.npz            arrays for record 0
#   00001.npz            arrays for record 1
#   …

arrays_list, metadata_list = load("output/my_run/")

print(metadata_list[0])
# {"index": 0, "run_id": "...", "input_len": 6, "num_layers": 13,
#  "properties": {"label": 1, "id": 0},
#  "spans": {"entity": {"start": 0, "end": 1}, "country": {"start": 5, "end": 6}},
#  "arrays": {"input_hidden_states_mean": [13, 768], "span_entity": [13, 768], ...}}

print(arrays_list[0]["input_hidden_states_mean"].shape)  # (13, 768)
print(arrays_list[0]["span_entity"].shape)               # (13, 768)
```

---

## Spans

Spans let you average hidden states over a specific sub-sequence of the prompt — for example a subject, a relation, a sentence, or a cue phrase — and store the result as a dedicated array alongside the full extraction.

### Span value types

All four types can be freely mixed within a single instance:

| Value | Resolved as | Notes |
|---|---|---|
| `"The cat"` | `TextSpan("The cat")` | Substring lookup via offset mapping |
| `TextSpan("The cat")` | Substring lookup | Supports `occurrence=`, `strip=` |
| `(0, 3)` | `SpanSpec(0, 3)` | Explicit token indices |
| `SpanSpec(0, 3)` | Token indices, verbatim | Supports negative indices |

### TextSpan — robust substring spans

`TextSpan` is the recommended way to define spans because it is stable across tokenizer versions, model families, and preprocessing changes. It uses `return_offsets_mapping=True` (available on all fast tokenizers) to map the substring's character positions to token indices at inference time.

```python
from internals_extraction import TextSpan, InternalsInstance

instance = InternalsInstance(
    text="The quick brown fox jumps over the lazy dog.",
    spans={
        "subject":  TextSpan("The quick brown fox"),
        "verb":     TextSpan("jumps"),
        "object":   TextSpan("the lazy dog"),
    },
)
```

### TextSpan with a label — list form

When you want the span definition to carry its own name (useful when building instances programmatically), use the `label` parameter and pass a list instead of a dict:

```python
instance = InternalsInstance(
    text="The quick brown fox jumps over the lazy dog.",
    spans=[
        TextSpan("The quick brown fox", label="subject"),
        TextSpan("jumps",               label="verb"),
        TextSpan("the lazy dog",        label="object"),
    ],
)
# instance.spans == {"subject": TextSpan(...), "verb": TextSpan(...), "object": TextSpan(...)}
```

The `label` becomes the dictionary key.  A `ValueError` is raised for missing or duplicate labels.

### Selecting a specific occurrence

When the same substring appears multiple times, use `occurrence` (0-indexed) to select which one:

```python
text = "the cat chased the dog because the cat was hungry"

instance = InternalsInstance(
    text=text,
    spans=[
        TextSpan("the cat", label="subject",   occurrence=0),   # first "the cat"
        TextSpan("the dog", label="object",    occurrence=0),
        TextSpan("the cat", label="subject_2", occurrence=1),   # second "the cat"
        TextSpan("the",     label="the_3rd",   occurrence=2),   # third "the"
    ],
)
```

A `SpanResolutionError` is raised if the requested occurrence does not exist, with a message showing how many occurrences were found.

### SpanSpec — explicit token indices

Use when you already know the token positions (e.g. from a pre-tokenised dataset with BIO tags):

```python
from internals_extraction import SpanSpec

spans={
    "cls":      SpanSpec(0, 1),    # token 0 only
    "sentence": SpanSpec(1, -1),   # tokens 1 to second-to-last (negative indexing)
    "last":     SpanSpec(-1, 0),   # would be empty — resolves to zero vector
}
```

---

## Logit-lens

Logit-lens applies the final LayerNorm + LM-head to each layer's hidden states, revealing what the model "would have predicted" at each depth. Computed entirely on CPU in the background thread.

```python
import internals_extraction
from transformers import AutoModelForCausalLM, AutoTokenizer

model     = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

internals_extraction.register_model(model)           # copy weights to CPU once
internals_extraction.config.extract_logit_lens = True

inputs = tokenizer("The capital of France is", return_tensors="pt")
model.generate(**inputs, max_new_tokens=1)

run = internals_extraction.get_latest()
# logit_lens[layer] → (batch, input_len, vocab)
for i, lens in enumerate(run.logit_lens):
    top = tokenizer.decode([lens[0, -1, :].argmax()])
    print(f"Layer {i:2d}: {top!r}")
```

---

## HuggingFace datasets integration

### Input — `from_hf_dataset()`

Build an `InternalsDataset` directly from a `datasets.Dataset`:

```python
from datasets import load_dataset
from internals_extraction import InternalsDataset, TextSpan

ds = load_dataset("sst2", split="validation[:200]")

dataset = InternalsDataset.from_hf_dataset(
    ds,
    text_col="sentence",           # column containing the input text
    property_cols=["label", "idx"],  # columns to keep as instance properties
    spans={"full": TextSpan("...")}, # optional: uniform spans for all rows
)

records = dataset.run(model, tokenizer, generate_kwargs={"max_new_tokens": 1})
```

#### Per-row spans from a column

If your dataset has a column of per-row span annotations:

```python
# Each row: {"text": "...", "spans": {"subject": "The cat", "verb": [2, 3]}}
dataset = InternalsDataset.from_hf_dataset(
    ds,
    text_col="text",
    spans_col="spans",   # per-row span dict; values may be strings, [start,end], or {"start":,"end":}
)
```

### Output — `to_hf_dataset()`

Convert records to a `datasets.Dataset` for saving or pushing to the Hub:

```python
from internals_extraction import to_hf_dataset

hf_ds = to_hf_dataset(records)
# Columns: label, idx, run_id, input_len, num_layers, num_output_tokens,
#          input_hidden_states_mean, output_hidden_states_mean, logits,
#          span_entity, span_entity_start, span_entity_end, ...

hf_ds.save_to_disk("my_internals/")
hf_ds.push_to_hub("my-org/gpt2-sst2-internals")
```

Each `span_{name}` column stores a `(num_layers, hidden)` nested list. Resolved token boundaries appear as `span_{name}_start` / `span_{name}_end` integer columns.

---

## Configuration

All options have sensible defaults and can be changed before inference:

```python
import internals_extraction

cfg = internals_extraction.config

# ── What to extract ──────────────────────────────────────────────────────────
cfg.extract_hidden_states     = True    # layer-wise hidden states  (default: True)
cfg.extract_attentions        = True    # attention weight matrices  (default: True)
cfg.extract_logits            = True    # per-step LM-head output    (default: True)
cfg.extract_logit_lens        = False   # logit-lens projection       (default: False)

# ── Aggregation ───────────────────────────────────────────────────────────────
cfg.aggregate_attention_heads = True    # mean across heads           (default: True)

# ── Storage ───────────────────────────────────────────────────────────────────
cfg.max_stored_runs           = 10      # ring-buffer capacity         (default: 10)

# ── Worker ────────────────────────────────────────────────────────────────────
cfg.worker_queue_maxsize      = 0       # 0 = unbounded queue          (default: 0)
```

Or use the bulk helper:

```python
internals_extraction.set_config(
    extract_logit_lens=True,
    aggregate_attention_heads=False,
    max_stored_runs=100,
)
```

---

## API reference

### Inference helpers

#### `get_latest() → InternalsRun | None`
Most recently completed run, or `None`.

#### `get_run(run_id: str) → InternalsRun | None`
Look up a run by UUID.

#### `get_all() → list[InternalsRun]`
All runs in the ring buffer, oldest first.

#### `wait_for_run(run_id: str, timeout: float = 30.0) → InternalsRun | None`
Block until the run is finalized. Use when you need the result immediately after `generate()` returns (the background thread may still be processing).

#### `register_model(model) → None`
Copy LM-head and final LayerNorm weights to CPU once. Required for `extract_logit_lens`. Supported architectures: GPT-2, LLaMA / Mistral / Gemma, GPT-NeoX / Pythia, Falcon, OPT, BLOOM.

#### `set_config(**kwargs) → None`
Update multiple config fields at once. Also resizes the ring buffer if `max_stored_runs` is included.

---

### `InternalsConfig`

```python
@dataclass
class InternalsConfig:
    extract_hidden_states:     bool = True
    extract_attentions:        bool = True
    extract_logits:            bool = True
    extract_logit_lens:        bool = False
    aggregate_attention_heads: bool = True
    max_stored_runs:           int  = 10
    worker_queue_maxsize:      int  = 0
```

---

### `InternalsRun`

| Attribute | Type | Shape | Description |
|---|---|---|---|
| `run_id` | `str` | — | UUID assigned at the start of `generate()` |
| `input_len` | `int` | — | Number of prompt tokens |
| `num_layers` | `int` | — | `num_transformer_blocks + 1` (includes embedding) |
| `num_output_tokens` | `int` | — | Tokens generated |
| `is_finalized` | `bool` | — | `True` once background thread has finished |
| `input_hidden_states` | `list[ndarray]` | `[L] → (batch, T_in, H)` | Full per-layer prompt hidden states |
| `output_hidden_states` | `list[ndarray]` | `[L] → (batch, T_out, H)` | Full per-layer output hidden states |
| `input_hidden_states_mean` | `ndarray` | `(L, batch, H)` | Mean over prompt positions, per layer |
| `output_hidden_states_mean` | `ndarray` | `(L, batch, H)` | Mean over output tokens, per layer |
| `attentions` | `list[list[ndarray]]` | `[step][L] → (batch, [heads,] Q, K)` | Per-step, per-layer attention weights |
| `logits` | `ndarray` | `(steps, batch, vocab)` | Last-position logits per step |
| `logit_lens` | `list[ndarray]` | `[L] → (batch, T_in, vocab)` | Logit-lens projections |

---

### `TextSpan`

```python
@dataclass
class TextSpan:
    text:       str
    label:      str | None = None    # span name when used in list form
    occurrence: int        = 0       # 0 = first match, 1 = second, …
    strip:      bool       = True    # strip whitespace before matching
```

---

### `SpanSpec`

```python
@dataclass
class SpanSpec:
    start: int    # inclusive; negative = offset from end
    end:   int    # exclusive; negative = offset from end
```

---

### `InternalsInstance`

```python
InternalsInstance(
    text:       str | list[int],
    properties: dict = {},
    spans:      dict | list[TextSpan] | None = None,
)
```

`spans` accepts:

- **dict** — keys are names, values are `str`, `TextSpan`, `SpanSpec`, or `(int, int)`
- **list** — `TextSpan` objects with `label` set; labels become keys

---

### `InternalsRecord`

| Attribute | Type | Description |
|---|---|---|
| `instance` | `InternalsInstance` | Original input |
| `run` | `InternalsRun` | Full extracted internals |
| `resolved_spans` | `dict[str, SpanSpec] \| None` | Spans after TextSpan → token-index resolution |
| `span_hidden_states_mean` | `dict[str, ndarray] \| None` | `{name: (num_layers, hidden)}` per span |
| `properties` | `dict` | Shortcut for `instance.properties` |
| `spans` | `dict` | Shortcut for `instance.spans` |

---

### `InternalsDataset`

```python
InternalsDataset(instances: list[InternalsInstance])

# Constructor
InternalsDataset.from_hf_dataset(
    dataset,
    text_col:      str           = "text",
    property_cols: list[str]     = None,   # None = all non-text columns
    spans:         dict          = None,   # uniform spans for every row
    spans_col:     str           = None,   # column with per-row span dicts
) → InternalsDataset

# Processing
.run(
    model,
    tokenizer,
    generate_kwargs:  dict  = {"max_new_tokens": 1},
    finalize_timeout: float = 60.0,
    verbose:          bool  = True,
) → list[InternalsRecord]
```

---

### Serialisation

#### `dump(records, outdir, save_attentions=False, save_full_hidden_states=False)`

Write to `outdir/`:

```
metadata.jsonl      # one JSON object per record
00000.npz           # arrays for record 0
00001.npz           # …
```

Each `.npz` contains (batch dimension squeezed):

| Key | Shape | Condition |
|---|---|---|
| `input_hidden_states_mean` | `(L, H)` | always |
| `output_hidden_states_mean` | `(L, H)` | always |
| `logits` | `(steps, vocab)` | always |
| `logit_lens` | `(L, T_in, vocab)` | if `extract_logit_lens=True` |
| `span_{name}` | `(L, H)` | per named span |
| `attentions_step{N}` | `(L, Q, K)` | if `save_attentions=True` |
| `input_hidden_states` | `(L, T_in, H)` | if `save_full_hidden_states=True` |
| `output_hidden_states` | `(L, T_out, H)` | if `save_full_hidden_states=True` |

`metadata.jsonl` fields per record:

```json
{
  "index": 0,
  "run_id": "...",
  "input_len": 7,
  "num_layers": 13,
  "num_output_tokens": 1,
  "properties": {"label": 1, "id": 0},
  "spans": {"entity": {"start": 0, "end": 1}},
  "arrays": {"input_hidden_states_mean": [13, 768], "span_entity": [13, 768]}
}
```

#### `load(outdir) → (arrays_list, metadata_list)`

```python
arrays_list, metadata_list = load("outdir/")

arrays_list[0]["input_hidden_states_mean"].shape   # (13, 768)
arrays_list[0]["span_entity"].shape                # (13, 768)
metadata_list[0]["properties"]                     # {"label": 1, "id": 0}
metadata_list[0]["spans"]["entity"]                # {"start": 0, "end": 1}
```

#### `to_hf_dataset(records, save_full_hidden_states=False) → datasets.Dataset`

Columns: all `properties` fields + `run_id`, `input_len`, `num_layers`, `num_output_tokens`, array columns (as nested lists), `span_{name}_start`, `span_{name}_end`.

---

## Architecture notes

### Thread model

The background worker is a **daemon thread** — not a subprocess — so there is no IPC serialization cost. CPU-bound work (numpy conversions, aggregation) releases the Python GIL, allowing true parallelism with the GPU-side `generate()` call.

### KV-cache awareness

- **Step 0 (prefill):** full sequence → shape `(batch, input_len, hidden)`
- **Steps 1..K (decode):** one new token → shape `(batch, 1, hidden)`

Hidden states are split at `input_len` and accumulated correctly regardless of whether KV-cache is enabled.

### TextSpan resolution

`_resolve_text_spans` calls `tokenizer(text, return_offsets_mapping=True)` once per instance to obtain character→token alignment, then maps each `TextSpan`'s character range. Special tokens with offset `(0, 0)` (BOS, EOS, PAD) are skipped automatically. Requires a HuggingFace **fast tokenizer**.

---

## Running tests

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

76 tests, no GPU or real model required.

---

## Supported model families

| Family | LM head | Final norm |
|---|---|---|
| GPT-2 | `lm_head` | `transformer.ln_f` |
| LLaMA / Mistral / Gemma | `lm_head` | `model.norm` |
| GPT-NeoX / Pythia | `embed_out` | `gpt_neox.final_layer_norm` |
| Falcon | `lm_head` | `transformer.ln_f` |
| OPT | `lm_head` | `model.decoder.final_layer_norm` |
| BLOOM | `lm_head` | `transformer.word_embeddings_layernorm` |

For logit-lens only. All other extractions work on any `PreTrainedModel`.

---

## License

MIT
