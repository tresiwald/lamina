"""
Concrete example: exploring GPT-2 internals with internals_extraction
====================================================================

Run:
    pip install transformers torch
    python examples/explore_gpt2.py

What this script demonstrates
------------------------------
1. Zero-config extraction — just import.
2. Hidden-state geometry: how much the representation changes layer by layer.
3. Attention patterns: which tokens attend to which at each layer.
4. Logit tracking: how the next-token distribution sharpens over steps.
5. Logit-lens: what GPT-2 "would have predicted" at each layer depth.
6. Input vs output token comparison: how their mean representations differ.
"""

import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


# ── Step 1: import the plugin ─────────────────────────────────────────────────
# This is the only line needed. No model changes, no wrappers.
import internals_extraction

from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Step 2: load model (unchanged) ───────────────────────────────────────────
print("Loading GPT-2 …")
MODEL = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model     = AutoModelForCausalLM.from_pretrained(MODEL)
model.eval()

# ── Step 3 (optional): enable logit-lens ─────────────────────────────────────
# Copy lm_head + LayerNorm weights to CPU once so the background thread
# can compute logit-lens projections without touching the GPU.
internals_extraction.register_model(model)
internals_extraction.config.extract_logit_lens        = True
internals_extraction.config.aggregate_attention_heads = True   # mean over heads

# ── Step 4: normal inference — nothing special needed ────────────────────────
PROMPT       = "The Eiffel Tower is located in the city of"
MAX_NEW      = 10

print(f"\nPrompt : {PROMPT!r}")
inputs  = tokenizer(PROMPT, return_tensors="pt")
input_ids = inputs["input_ids"]

output_ids = model.generate(
    **inputs,
    max_new_tokens=MAX_NEW,
    do_sample=False,          # greedy for reproducibility
)
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"Output : {generated_text!r}\n")

# ── Step 5: retrieve internals ────────────────────────────────────────────────
run = internals_extraction.get_latest()
print(run)
print()

# Convenience aliases
input_len  = run.input_len
num_layers = run.num_layers          # 13 for GPT-2 (embed + 12 blocks)
vocab_size = run.logits.shape[-1]    # 50257
input_tokens  = tokenizer.convert_ids_to_tokens(input_ids[0])
output_tokens = tokenizer.convert_ids_to_tokens(
    output_ids[0, input_len:]
)

sep = "─" * 70


# ════════════════════════════════════════════════════════════════════════════
# 1. HIDDEN-STATE GEOMETRY — cosine similarity between consecutive layers
# ════════════════════════════════════════════════════════════════════════════
print(sep)
print("1. HIDDEN-STATE GEOMETRY — layer-to-layer cosine similarity")
print("   (mean representation of the input tokens)")
print(sep)

# input_hidden_states_mean: (num_layers, batch=1, hidden=768)
means = run.input_hidden_states_mean[:, 0, :]   # → (num_layers, 768)

for i in range(num_layers - 1):
    a, b = means[i], means[i + 1]
    cos  = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    bar  = "█" * int(cos * 30)
    print(f"  Layer {i:2d} → {i+1:2d}  cos={cos:.4f}  {bar}")

print()


# ════════════════════════════════════════════════════════════════════════════
# 2. INPUT vs OUTPUT TOKEN REPRESENTATIONS
# ════════════════════════════════════════════════════════════════════════════
print(sep)
print("2. INPUT vs OUTPUT — L2 distance between their mean HS per layer")
print(sep)

# input_hidden_states_mean:  (num_layers, 1, hidden)
# output_hidden_states_mean: (num_layers, 1, hidden)
in_means  = run.input_hidden_states_mean[:, 0, :]   # (num_layers, hidden)
out_means = run.output_hidden_states_mean[:, 0, :]  # (num_layers, hidden)

for i in range(num_layers):
    dist = np.linalg.norm(in_means[i] - out_means[i])
    bar  = "█" * min(int(dist / 5), 40)
    print(f"  Layer {i:2d}  ‖in − out‖ = {dist:7.2f}  {bar}")

print()


# ════════════════════════════════════════════════════════════════════════════
# 3. ATTENTION PATTERNS — top attended-to token per layer (step 0)
# ════════════════════════════════════════════════════════════════════════════
print(sep)
print("3. ATTENTION PATTERNS — where does the *last* prompt token attend?")
print(f"   (step 0, head-averaged, query = last prompt token = {input_tokens[-1]!r})")
print(sep)

# attentions[step][layer] → (batch, seq_q, seq_k)
step0_attentions = run.attentions[0]   # list of num_layers arrays

for layer_idx, attn in enumerate(step0_attentions):
    row = attn[0, -1, :]               # last query position, shape (seq_k,)
    top_k = 3
    top_indices = row.argsort()[-top_k:][::-1]
    tops = [
        f"{input_tokens[j]!r}({row[j]:.2f})"
        for j in top_indices
        if j < len(input_tokens)
    ]
    print(f"  Layer {layer_idx:2d}  →  {', '.join(tops)}")

print()


# ════════════════════════════════════════════════════════════════════════════
# 4. LOGIT TRACKING — how the top-1 prediction evolves across generation steps
# ════════════════════════════════════════════════════════════════════════════
print(sep)
print("4. LOGIT TRACKING — top-1 next-token prediction at each step")
print(sep)

# logits: (num_steps, batch=1, vocab)
print(f"  {'Step':>4}  {'Generated token':>20}  {'Top prediction':>20}  {'Prob':>6}")
print(f"  {'────':>4}  {'───────────────':>20}  {'──────────────':>20}  {'────':>6}")

for step_idx in range(run.logits.shape[0]):
    step_logits = run.logits[step_idx, 0]       # (vocab,)
    probs       = _softmax(step_logits)
    top_id      = probs.argmax()
    top_token   = tokenizer.decode([top_id])
    top_prob    = probs[top_id]
    gen_token   = output_tokens[step_idx] if step_idx < len(output_tokens) else "—"
    print(f"  {step_idx:>4}  {gen_token!r:>20}  {top_token!r:>20}  {top_prob:>6.3f}")

print()


# ════════════════════════════════════════════════════════════════════════════
# 5. LOGIT-LENS — what does each layer "predict" for the last prompt position?
# ════════════════════════════════════════════════════════════════════════════
print(sep)
print("5. LOGIT-LENS — top-3 predictions per layer at the last prompt position")
print(f"   (position = {input_tokens[-1]!r}, first correct answer would be 'Paris')")
print(sep)

# logit_lens[layer] → (batch, input_len, vocab)
if run.logit_lens is not None:
    print(f"  {'Layer':>6}  Top-3 predictions")
    print(f"  {'─────':>6}  ─────────────────")
    for layer_idx, lens in enumerate(run.logit_lens):
        last_logits = lens[0, -1, :]              # last prompt position
        probs       = _softmax(last_logits)
        top3_ids    = probs.argsort()[-3:][::-1]
        top3        = [
            f"{tokenizer.decode([t])!r}({probs[t]:.3f})"
            for t in top3_ids
        ]
        marker = "  ◀ answer!" if any("Paris" in tokenizer.decode([t]) for t in top3_ids) else ""
        print(f"  {layer_idx:>6}  {', '.join(top3)}{marker}")
else:
    print("  (logit_lens not available)")

print()


# ════════════════════════════════════════════════════════════════════════════
# 6. RAW SHAPES — quick inventory of everything captured
# ════════════════════════════════════════════════════════════════════════════
print(sep)
print("6. RAW SHAPES — inventory of captured arrays")
print(sep)

print(f"  input_hidden_states      : {len(run.input_hidden_states)} layers, "
      f"each {run.input_hidden_states[0].shape}")
print(f"  output_hidden_states     : {len(run.output_hidden_states)} layers, "
      f"each {run.output_hidden_states[0].shape}")
print(f"  input_hidden_states_mean : {run.input_hidden_states_mean.shape}")
print(f"  output_hidden_states_mean: {run.output_hidden_states_mean.shape}")
print(f"  attentions               : {len(run.attentions)} steps × "
      f"{len(run.attentions[0])} layers, "
      f"step0/layer0 = {run.attentions[0][0].shape}")
print(f"  logits                   : {run.logits.shape}")
if run.logit_lens:
    print(f"  logit_lens               : {len(run.logit_lens)} layers, "
          f"each {run.logit_lens[0].shape}")
print()
