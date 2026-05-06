# lamina

> *"It is the little grey cells, mon ami."* — Hercule Poirot

**lamina** is a modular library for extracting and examining the internal representations of neural language models — hidden states, attentions, logits, and logit-lens projections — **without modifying model code** and **without slowing down GPU inference**.

```
pip install lamina[hf]
```

---

## What it extracts

| Signal | Shape | Description |
|---|---|---|
| `input_hidden_states` | `list[layer] → (batch, input_len, hidden)` | Hidden states for every prompt token, per layer |
| `output_hidden_states` | `list[layer] → (batch, output_tokens, hidden)` | Hidden states for every generated token, per layer |
| `input_hidden_states_mean` | `(num_layers, batch, hidden)` | Per-layer mean over the prompt sequence |
| `output_hidden_states_mean` | `(num_layers, batch, hidden)` | Per-layer mean over generated tokens |
| `encoder_hidden_states` | `list[layer] → (batch, enc_len, hidden)` | Encoder representations (T5, BART, …) |
| `attentions` | `list[step][layer] → (batch, [heads,] Q, K)` | Per-layer attention weights |
| `logits` | `(steps, batch, vocab)` | LM-head output at the last position of each step |
| `logit_lens` | `list[layer] → (batch, input_len, vocab)` | LM-head applied to each layer's hidden state *(opt-in)* |
| `span_<name>` | `(num_layers, hidden)` | Per-layer mean over a named token span *(opt-in)* |

Layer `0` is always the embedding output; layers `1..N` are transformer block outputs.

---

## Architecture

```
lamina
│
├── core/               Library-agnostic: InternalsConfig, InternalsRun,
│                       InternalsStore (ring buffer), BackgroundWorker
│
├── extractors/         Framework-specific capture
│   ├── hf/             HuggingFace Transformers — generate() + run_forward()
│   └── vllm/           vLLM (planned)
│
├── interventions/      Active model modifications (planned)
│   ├── ActivationPatch     replace a layer's output with a stored tensor
│   ├── AttentionMask       zero-out / scale specific heads or positions
│   └── SteeringVector      add a steering vector to the residual stream
│
├── backends/           Pluggable storage
│   ├── filesystem      .npz + metadata.jsonl  (default)
│   ├── hf_dataset      HuggingFace datasets.Dataset
│   ├── redis           (planned)
│   └── mongodb         (planned)
│
└── applications/       Higher-level APIs
    ├── dataset         InternalsDataset / InternalsInstance / InternalsRecord
    └── spans           TextSpan, SpanSpec, offset-mapping resolver
```

### How inference capture works

```
  model.generate()
       │
       ▼ lamina.extractors.hf — installed on import
  ┌──────────────────────────────────────────────────────────┐
  │  ① forces output_hidden_states / output_attentions       │
  │  ② registers register_forward_hook on the model instance │
  │  ③ calls original generate  ← GPU runs at full speed     │
  │                                                          │
  │  per-step hook (fires after each forward pass):          │
  │    • grabs tensor references  ← nanoseconds              │
  │    • enqueues to background worker  ← non-blocking       │
  └──────────────────────────────────────────────────────────┘
       │  queue
       ▼
  ┌──────────────────────────────────────────────────────────┐
  │  BackgroundWorker (daemon thread)                        │
  │  • tensor.detach().cpu().numpy()  ← GPU freed per step   │
  │  • aggregates hidden states / attentions / logits        │
  │  • finalises InternalsRun into thread-safe ring buffer   │
  └──────────────────────────────────────────────────────────┘
       │
       ▼
  lamina.get_latest()   ← your code
```

**GPU memory:** hidden-state tensors are materialised during the forward pass (unavoidable), then immediately moved to CPU in the background thread. Peak overhead = one generation step's worth of hidden states.

---

## Installation

```bash
# HuggingFace Transformers (most common)
pip install lamina[hf]

# With HuggingFace datasets backend
pip install lamina[hf,backends]

# Everything
pip install lamina[all]

# Development
pip install lamina[dev]
```

**Core** (`pip install lamina`) requires only `numpy` and has no ML-framework dependency — useful for analysis scripts that only read saved data.

| Extra | Adds |
|---|---|
| `[hf]` | `transformers>=4.30` |
| `[backends]` | `datasets>=2.0` (HFDatasetBackend) |
| `[interventions]` | `torch` (explicit dep for tensor mutation) |
| `[vllm]` | `vllm` *(planned)* |
| `[all]` | all of the above |

---

## Quick start

### Decoder-only model (GPT-2, LLaMA, …)

```python
import lamina
from transformers import AutoModelForCausalLM, AutoTokenizer

model     = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

inputs = tokenizer("The Eiffel Tower is located in", return_tensors="pt")
model.generate(**inputs, max_new_tokens=20)

run = lamina.get_latest()
print(run)
# InternalsRun(run_id='...', input_len=7, num_output_tokens=20,
#              num_layers=13, finalized=True)

print(run.input_hidden_states_mean.shape)   # (13, 1, 768)
print(run.output_hidden_states_mean.shape)  # (13, 1, 768)
print(run.input_hidden_states[6].shape)     # (1, 7, 768)
print(run.attentions[0][0].shape)           # (1, 7, 7)
print(run.logits.shape)                     # (20, 1, 50257)
```

### Encoder-decoder model (T5, BART, …)

```python
model     = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
tokenizer = AutoTokenizer.from_pretrained("t5-small")

inputs = tokenizer("translate English to French: Hello world", return_tensors="pt")
model.generate(**inputs, max_new_tokens=10)

run = lamina.get_latest()
print(run.is_encoder_decoder)               # True
print(run.encoder_hidden_states_mean.shape) # (encoder_layers, 1, 512)
print(run.output_hidden_states_mean.shape)  # (decoder_layers, 1, 512)
```

### Encoder-only model (BERT, RoBERTa, …)

```python
from lamina import run_forward

model     = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

inputs = tokenizer("The movie was great!", return_tensors="pt")
run_forward(model, **inputs)

run = lamina.get_latest()
print(run.input_hidden_states_mean.shape)   # (13, 1, 768)
print(run.logits.shape)                     # (1, 1, 2)  ← (steps, batch, num_labels)
```

---

## Supported model types

| AutoModel class | Inference call | Notes |
|---|---|---|
| `ForCausalLM`, `LMHeadModel` | `generate()` | Decoder-only |
| `ForSeq2SeqLM`, `ForConditionalGeneration` | `generate()` | Encoder-decoder; `encoder_hidden_states` available |
| `ForSpeechSeq2Seq`, `ForVision2Seq` | `generate()` | |
| `ForSequenceClassification` | `run_forward()` | 2-D logits `(batch, num_labels)` |
| `ForTokenClassification` | `run_forward()` | 3-D logits `(batch, seq, num_labels)` |
| `ForQuestionAnswering` | `run_forward()` | `start_logits`/`end_logits` merged → `(batch, 2, seq)` |
| `ForMaskedLM` | `run_forward()` | Full-sequence vocab logits |
| `ForMultipleChoice`, `ForNextSentencePrediction` | `run_forward()` | |
| `ForImageClassification`, `ForAudioClassification` | `run_forward()` | |

Detection uses `model.can_generate()` (transformers ≥ 4.18) with class-name and config-level heuristic fallbacks.

---

## Dataset processing

### InternalsInstance & InternalsDataset

```python
from lamina import InternalsDataset, InternalsInstance, TextSpan

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
    print(rec.properties)
    print(rec.run.input_hidden_states_mean.shape)
    if rec.span_hidden_states_mean:
        print(rec.span_hidden_states_mean["entity"].shape)  # (num_layers, hidden)
    print(rec.resolved_spans)   # {"entity": SpanSpec(0,1), "country": SpanSpec(5,6)}
```

### From a HuggingFace dataset

```python
from datasets import load_dataset

ds = load_dataset("sst2", split="validation[:200]")

dataset = InternalsDataset(
    ds,
    text_col="sentence",
    property_cols=["label", "idx"],
)
```

Or equivalently via the classmethod:

```python
dataset = InternalsDataset.from_hf_dataset(ds, text_col="sentence", property_cols=["label"])
```

---

## Spans

Spans average hidden states over a named sub-sequence of the prompt, stored alongside the full extraction.

### Span value types

| Value | Resolved as |
|---|---|
| `"The cat"` | `TextSpan("The cat")` — substring via offset mapping |
| `TextSpan("the cat", occurrence=1)` | second occurrence of "the cat" |
| `(0, 3)` | `SpanSpec(0, 3)` — explicit token indices |
| `SpanSpec(1, -1)` | tokens 1 to second-to-last |

```python
instance = InternalsInstance(
    text="The quick brown fox jumps over the lazy dog.",
    spans=[
        TextSpan("The quick brown fox", label="subject"),
        TextSpan("jumps",               label="verb"),
        TextSpan("the lazy dog",        label="object"),
    ],
)
```

---

## Storage backends

### Filesystem (default)

```python
from lamina.backends import dump, load

dump(records, "output/my_run/")
# output/my_run/
#   metadata.jsonl
#   00000.npz
#   00001.npz

arrays_list, metadata_list = load("output/my_run/")
print(arrays_list[0]["input_hidden_states_mean"].shape)  # (13, 768)
print(metadata_list[0]["properties"])                    # {"label": 1, "id": 0}
```

Or via the backend class:

```python
from lamina.backends.filesystem import FilesystemBackend

backend = FilesystemBackend("output/", save_attentions=True)
backend.write(records)
arrays_list, meta_list = FilesystemBackend.read("output/")
```

### HuggingFace datasets

```python
from lamina.backends import to_hf_dataset

hf_ds = to_hf_dataset(records)
hf_ds.save_to_disk("my_internals/")
hf_ds.push_to_hub("my-org/gpt2-sst2-internals")
```

---

## Interventions *(planned)*

Interventions modify tensors during the forward pass and compose freely with extraction — you get internals from an intervened-upon run:

```python
from lamina.interventions import ActivationPatch, AttentionMask, SteeringVector

# Replace layer 12's output with a cached activation
with ActivationPatch(layer=12, value=source_run.input_hidden_states[12])(model):
    run_forward(model, **inputs)
    patched_run = lamina.get_latest()

# Zero out heads 0–3 of layer 8
with AttentionMask(layer=8, heads=[0, 1, 2, 3])(model):
    model.generate(input_ids, max_new_tokens=20)

# Steer generation via a direction in residual space
with SteeringVector(layer=16, vector=direction, scale=20.0)(model):
    model.generate(input_ids, max_new_tokens=50)
```

The `Intervention` ABC is fully defined; concrete implementations are coming in the next release.

---

## Logit-lens

```python
import lamina
from transformers import AutoModelForCausalLM, AutoTokenizer

model     = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

lamina.register_model(model)            # copy weights to CPU once
lamina.config.extract_logit_lens = True

inputs = tokenizer("The capital of France is", return_tensors="pt")
model.generate(**inputs, max_new_tokens=1)

run = lamina.get_latest()
for i, lens in enumerate(run.logit_lens):   # lens: (batch, input_len, vocab)
    top = tokenizer.decode([lens[0, -1, :].argmax()])
    print(f"Layer {i:2d}: {top!r}")
```

Supported architectures for logit-lens: GPT-2, LLaMA/Mistral/Gemma, GPT-NeoX/Pythia, Falcon, OPT, BLOOM.

---

## Configuration

```python
import lamina

lamina.config.extract_hidden_states     = True    # default: True
lamina.config.extract_attentions        = True    # default: True
lamina.config.extract_logits            = True    # default: True
lamina.config.extract_logit_lens        = False   # default: False — needs register_model()
lamina.config.aggregate_attention_heads = True    # default: True — mean across heads
lamina.config.max_stored_runs           = 10      # default: 10
lamina.config.worker_queue_maxsize      = 0       # default: 0 — unbounded

# Or bulk update:
lamina.set_config(extract_logit_lens=True, max_stored_runs=100)
```

---

## API reference

### Top-level helpers

| Function | Returns | Description |
|---|---|---|
| `lamina.get_latest()` | `InternalsRun \| None` | Most recent completed run |
| `lamina.get_run(run_id)` | `InternalsRun \| None` | Look up by UUID |
| `lamina.get_all()` | `list[InternalsRun]` | All runs in the ring buffer |
| `lamina.wait_for_run(id, timeout)` | `InternalsRun \| None` | Block until finalised |
| `lamina.run_forward(model, **inputs)` | `ModelOutput` | Capture from a single forward pass |
| `lamina.register_model(model)` | `None` | Copy LM-head to CPU for logit-lens |
| `lamina.set_config(**kw)` | `None` | Update config fields |

### `InternalsRun` attributes

| Attribute | Shape | Description |
|---|---|---|
| `input_hidden_states` | `[L] → (batch, T_in, H)` | Prompt hidden states, per layer |
| `output_hidden_states` | `[L] → (batch, T_out, H)` | Generated-token hidden states |
| `encoder_hidden_states` | `[L] → (batch, T_enc, H)` | Encoder representations (enc-dec models) |
| `input_hidden_states_mean` | `(L, batch, H)` | Mean over prompt positions |
| `output_hidden_states_mean` | `(L, batch, H)` | Mean over output tokens |
| `encoder_hidden_states_mean` | `(L, batch, H)` | Mean over encoder positions |
| `attentions` | `[step][L] → (batch, [heads,] Q, K)` | Attention weights |
| `encoder_attentions` | `[L] → (batch, [heads,] Q, K)` | Encoder self-attentions |
| `logits` | `(steps, batch, vocab)` | LM-head logits per step |
| `logit_lens` | `[L] → (batch, T_in, vocab)` | Logit-lens projections |
| `run_id` | `str` | UUID |
| `input_len` | `int` | Prompt length in tokens |
| `num_layers` | `int` | Transformer blocks + 1 (embedding) |
| `is_encoder_decoder` | `bool` | True for T5, BART, … |
| `is_finalized` | `bool` | True once the worker thread has finished |

---

## Running tests

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

No GPU or real model required — all tests use lightweight fake models.

---

## Notebooks

| Notebook | Topics |
|---|---|
| [`01_quickstart.ipynb`](notebooks/01_quickstart.ipynb) | Import, run, inspect all signals |
| [`02_spans.ipynb`](notebooks/02_spans.ipynb) | TextSpan, SpanSpec, occurrence, list form |
| [`03_dataset_processing.ipynb`](notebooks/03_dataset_processing.ipynb) | InternalsDataset, dump/load, properties |
| [`04_hf_datasets.ipynb`](notebooks/04_hf_datasets.ipynb) | from_hf_dataset, to_hf_dataset, push_to_hub |

---

## License

MIT
