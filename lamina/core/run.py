"""
InternalsRun — data container for one inference run.

Created when inference starts, incrementally filled by the background
worker thread, then finalised when inference returns.
"""
from __future__ import annotations

import threading
from typing import Dict, List, Optional

import numpy as np


class InternalsRun:
    """
    All extracted internals for one ``model.generate()`` (or forward) call.

    Attributes set *after* ``_finalize()`` has been called
    -------------------------------------------------------
    input_hidden_states : list[np.ndarray] | None
        Per-layer hidden states of the **input** sequence.
        Shape per layer: ``(batch, input_len, hidden_dim)``.
        Layer 0 = embedding output, layers 1..N = transformer block outputs.

        * Decoder-only: prompt token representations (generate step 0).
        * Encoder-decoder: encoder representations (aliased from
          ``encoder_hidden_states``).
        * Encoder-only: full-sequence representations (single forward pass).

    output_hidden_states : list[np.ndarray] | None
        Per-layer hidden states of the **generated tokens**.
        Shape per layer: ``(batch, num_output_tokens, hidden_dim)``.
        None for encoder-only models.

    encoder_hidden_states : list[np.ndarray] | None
        Encoder representations for encoder-decoder models (T5, BART, …).
        Shape per layer: ``(batch, input_len, hidden_dim)``.
        None for decoder-only and encoder-only models.

    input_hidden_states_mean : np.ndarray | None
        ``(num_layers, batch, hidden_dim)`` — mean over the sequence axis.

    output_hidden_states_mean : np.ndarray | None
        ``(num_layers, batch, hidden_dim)`` — mean over the token axis.

    encoder_hidden_states_mean : np.ndarray | None
        ``(num_encoder_layers, batch, hidden_dim)``.
        None for decoder-only and encoder-only models.

    attentions : list[list[np.ndarray]] | None
        ``attentions[step][layer]`` → ``(batch, [heads,] seq_q, seq_k)``

    encoder_attentions : list[np.ndarray] | None
        Encoder self-attentions for encoder-decoder models (step 0).

    logits : np.ndarray | None
        ``(num_output_tokens, batch, vocab_size)``

    logit_lens : list[np.ndarray] | None
        ``logit_lens[layer]`` → ``(batch, input_len, vocab_size)``
        Only populated when ``extract_logit_lens=True``.
    """

    __slots__ = (
        # identity / meta
        "run_id",
        "input_len",
        "_config",
        "_thinking_end_token_id",
        # step accumulation
        "_steps",
        "_lock",
        "_finalized",
        "_step_count",
        # streaming output state (used when extract_output_hidden_states=False)
        "_out_hs_sum",    # List[np.ndarray(hidden,)] per layer — running sum
        "_out_hs_count",  # int — number of output tokens summed
        "_out_last_hs",   # List[np.ndarray(hidden,)] per layer — last token
        # in-worker thinking-token detection state
        "_last_logits_argmax",   # int | None — argmax of previous step's logits
        "_thinking_hs_buffer",   # List[np.ndarray(hidden,)] | None
        "_thinking_hs_pos",      # int | None — output sequence position
        # public outputs
        "input_hidden_states",
        "output_hidden_states",
        "encoder_hidden_states",
        "input_hidden_states_mean",
        "output_hidden_states_mean",
        "encoder_hidden_states_mean",
        "last_output_hidden_state",
        "thinking_end_hidden_state",
        "thinking_end_token_pos",
        "attentions",
        "encoder_attentions",
        "logits",
        "logit_lens",
    )

    def __init__(
        self,
        run_id: str,
        input_len: int,
        config=None,                       # InternalsConfig | None
        thinking_end_token_id: Optional[int] = None,
    ) -> None:
        self.run_id: str = run_id
        self.input_len: int = input_len
        self._config = config
        self._thinking_end_token_id = thinking_end_token_id
        self._steps: List[Dict] = []
        self._lock = threading.Lock()
        self._finalized: bool = False
        self._step_count: int = 0

        # streaming output state
        self._out_hs_sum: Optional[List[np.ndarray]] = None
        self._out_hs_count: int = 0
        self._out_last_hs: Optional[List[np.ndarray]] = None

        # thinking detection state
        self._last_logits_argmax: Optional[int] = None
        self._thinking_hs_buffer: Optional[List[np.ndarray]] = None
        self._thinking_hs_pos: Optional[int] = None

        # public outputs
        self.input_hidden_states: Optional[List[np.ndarray]] = None
        self.output_hidden_states: Optional[List[np.ndarray]] = None
        self.encoder_hidden_states: Optional[List[np.ndarray]] = None
        self.input_hidden_states_mean: Optional[np.ndarray] = None
        self.output_hidden_states_mean: Optional[np.ndarray] = None
        self.encoder_hidden_states_mean: Optional[np.ndarray] = None
        self.last_output_hidden_state: Optional[np.ndarray] = None
        self.thinking_end_hidden_state: Optional[np.ndarray] = None
        self.thinking_end_token_pos: Optional[int] = None
        self.attentions: Optional[List[List[np.ndarray]]] = None
        self.encoder_attentions: Optional[List[np.ndarray]] = None
        self.logits: Optional[np.ndarray] = None
        self.logit_lens: Optional[List[np.ndarray]] = None

    # ------------------------------------------------------------------
    # Internal API (called from worker thread)
    # ------------------------------------------------------------------

    def _add_step(self, step_data: Dict) -> None:
        with self._lock:
            current_step = self._step_count
            self._step_count += 1
            streaming = (
                self._config is not None
                and not self._config.extract_output_hidden_states
            )

            # Step 0 always goes into _steps (needed for input / encoder hs)
            if not streaming or current_step == 0:
                self._steps.append(step_data)

        # ── Streaming output accumulation (steps 1+ when not storing full hs) ──
        if streaming and current_step > 0:
            hs_list = step_data.get("hidden_states")
            if hs_list is not None:
                # Each hs: (batch, seq, hidden); take last token, batch 0
                vecs = [hs[0, -1, :].astype(np.float32) for hs in hs_list]
                with self._lock:
                    if self._out_hs_sum is None:
                        self._out_hs_sum = [v.copy() for v in vecs]
                    else:
                        for i, v in enumerate(vecs):
                            self._out_hs_sum[i] += v
                    self._out_hs_count += 1
                    self._out_last_hs = [v.copy() for v in vecs]

        # ── In-worker thinking-token detection ───────────────────────────────
        # We detect at step k+1: if the previous step's logits predicted the
        # thinking end token (greedily), the *current* step processes that
        # token as input, so current hs is the representation we want.
        if self._thinking_end_token_id is not None and current_step > 0:
            if self._last_logits_argmax == self._thinking_end_token_id:
                hs_list = step_data.get("hidden_states")
                if hs_list is not None:
                    vecs = [hs[0, -1, :].astype(np.float32) for hs in hs_list]
                    with self._lock:
                        # Keep overwriting — we want the *last* occurrence
                        self._thinking_hs_buffer = vecs
                        self._thinking_hs_pos = current_step - 1  # output index

        # Update logits argmax for next step's thinking detection
        if self._thinking_end_token_id is not None:
            logits = step_data.get("logits")
            if logits is not None:
                try:
                    arr = logits[0] if logits.ndim == 2 else logits.ravel()
                    with self._lock:
                        self._last_logits_argmax = int(arr.argmax())
                except Exception:
                    pass

    def _finalize(
        self,
        lm_head_weight: Optional[np.ndarray],
        lm_head_bias: Optional[np.ndarray],
        final_norm_fn,
    ) -> None:
        """
        Aggregate per-step data into user-visible arrays.
        Called once, from the worker thread.

        When ``config.extract_output_hidden_states`` is False, ``output_hidden_states``
        is not built.  Only ``output_hidden_states_mean`` and
        ``last_output_hidden_state`` are populated from the streaming state
        accumulated in ``_add_step``.
        """
        with self._lock:
            if self._finalized:
                return
            steps = list(self._steps)
            streaming = (
                self._config is not None
                and not self._config.extract_output_hidden_states
            )

        # ── Detect model type ─────────────────────────────────────────────────
        enc_hs_source = next(
            (step["encoder_hidden_states"]
             for step in steps
             if step.get("encoder_hidden_states") is not None),
            None,
        )
        is_encoder_decoder = enc_hs_source is not None

        # ── Hidden states ─────────────────────────────────────────────────────
        all_have_hs = steps and (
            steps[0].get("hidden_states") is not None or is_encoder_decoder
        )
        if all_have_hs:
            if is_encoder_decoder:
                # ── Encoder-decoder ───────────────────────────────────────────
                enc_hs: List[np.ndarray] = list(enc_hs_source)
                self.encoder_hidden_states = enc_hs
                self.input_hidden_states = enc_hs

                self.encoder_hidden_states_mean = np.stack(
                    [hs.mean(axis=1) for hs in enc_hs], axis=0
                )
                self.input_hidden_states_mean = self.encoder_hidden_states_mean

                if streaming:
                    # Steps 1+ were streamed; step 0 decoder hs are in _steps[0]
                    # Include step 0's decoder hs in the streaming sum
                    dec_hs_0 = steps[0].get("hidden_states")
                    if dec_hs_0 is not None:
                        vecs = [hs[0, -1, :].astype(np.float32) for hs in dec_hs_0]
                        if self._out_hs_sum is None:
                            self._out_hs_sum = [v.copy() for v in vecs]
                            self._out_last_hs = [v.copy() for v in vecs]
                        else:
                            for i, v in enumerate(vecs):
                                self._out_hs_sum[i] += v
                        self._out_hs_count += 1
                    self._finalize_streaming_output()
                else:
                    dec_num_layers = (
                        len(steps[0]["hidden_states"])
                        if steps[0].get("hidden_states") is not None else 0
                    )
                    out_hs_per_layer: List[List[np.ndarray]] = [
                        [] for _ in range(dec_num_layers)
                    ]
                    for step in steps:
                        hs_list = step.get("hidden_states")
                        if hs_list is None:
                            continue
                        for layer_idx, hs in enumerate(hs_list):
                            out_hs_per_layer[layer_idx].append(hs[:, -1:, :])
                    out_hs: List[np.ndarray] = []
                    for layer_chunks in out_hs_per_layer:
                        if layer_chunks:
                            out_hs.append(np.concatenate(layer_chunks, axis=1))
                        else:
                            b = enc_hs[0].shape[0] if enc_hs else 1
                            h = enc_hs[0].shape[2] if enc_hs else 0
                            out_hs.append(np.empty((b, 0, h), dtype=np.float32))
                    self.output_hidden_states = out_hs
                    self._finalize_full_output_mean(out_hs)
                    self._finalize_last_output(out_hs)

            else:
                # ── Decoder-only / encoder-only ───────────────────────────────
                num_layers = len(steps[0]["hidden_states"])
                inp_hs: List[np.ndarray] = []

                for layer_idx, hs in enumerate(steps[0]["hidden_states"]):
                    inp_hs.append(hs[:, :self.input_len, :])

                self.input_hidden_states = inp_hs
                self.input_hidden_states_mean = (
                    np.stack([hs.mean(axis=1) for hs in inp_hs], axis=0)
                    if inp_hs else None
                )

                if streaming:
                    # Check if step 0 contains output tokens (no-KV-cache models)
                    step0_hs = steps[0]["hidden_states"]
                    if step0_hs[0].shape[1] > self.input_len:
                        vecs = [hs[0, self.input_len, :].astype(np.float32)
                                for hs in step0_hs]
                        if self._out_hs_sum is None:
                            self._out_hs_sum = [v.copy() for v in vecs]
                            self._out_last_hs = [v.copy() for v in vecs]
                        else:
                            for i, v in enumerate(vecs):
                                self._out_hs_sum[i] += v
                        self._out_hs_count += 1
                    self._finalize_streaming_output()
                else:
                    _out_hs_per_layer: List[List[np.ndarray]] = [
                        [] for _ in range(num_layers)
                    ]
                    for step_idx, step in enumerate(steps):
                        hs_list = step["hidden_states"]
                        if step_idx == 0:
                            for layer_idx, hs in enumerate(hs_list):
                                if hs.shape[1] > self.input_len:
                                    _out_hs_per_layer[layer_idx].append(
                                        hs[:, self.input_len:, :]
                                    )
                        else:
                            for layer_idx, hs in enumerate(hs_list):
                                _out_hs_per_layer[layer_idx].append(hs[:, -1:, :])

                    _out_hs: List[np.ndarray] = []
                    for layer_chunks in _out_hs_per_layer:
                        if layer_chunks:
                            _out_hs.append(np.concatenate(layer_chunks, axis=1))
                        else:
                            b = inp_hs[0].shape[0] if inp_hs else 1
                            h = inp_hs[0].shape[2] if inp_hs else 0
                            _out_hs.append(np.empty((b, 0, h), dtype=np.float32))
                    self.output_hidden_states = _out_hs
                    self._finalize_full_output_mean(_out_hs)
                    self._finalize_last_output(_out_hs)

            # ── Logit lens ────────────────────────────────────────────────────
            inp_hs_for_lens = self.input_hidden_states
            if (lm_head_weight is not None and final_norm_fn is not None
                    and inp_hs_for_lens):
                lens_list: List[np.ndarray] = []
                for hs in inp_hs_for_lens:
                    normed = final_norm_fn(hs)
                    logit = normed @ lm_head_weight.T
                    if lm_head_bias is not None:
                        logit = logit + lm_head_bias
                    lens_list.append(logit)
                self.logit_lens = lens_list

        # ── Attentions ────────────────────────────────────────────────────────
        if steps and steps[0].get("attentions") is not None:
            self.attentions = [step["attentions"] for step in steps]

        enc_att_source = next(
            (step["encoder_attentions"]
             for step in steps
             if step.get("encoder_attentions") is not None),
            None,
        )
        if enc_att_source is not None:
            self.encoder_attentions = list(enc_att_source)

        # ── Logits ───────────────────────────────────────────────────────────
        logit_list = []
        for step in steps:
            lg = step.get("logits")
            if lg is None:
                continue
            mode = step.get("logit_mode", "last_token")
            if mode == "last_token" and lg.ndim == 3:
                lg = lg[:, -1, :]
            logit_list.append(lg)
        if logit_list:
            try:
                self.logits = np.stack(logit_list, axis=0)
            except ValueError:
                self.logits = logit_list[0][np.newaxis]

        # ── Thinking end hidden state (from in-worker detection) ──────────────
        if self._thinking_hs_buffer is not None:
            self.thinking_end_hidden_state = np.stack(
                self._thinking_hs_buffer, axis=0
            )   # (num_layers, hidden_dim)
            self.thinking_end_token_pos = self._thinking_hs_pos

        # ── Clear per-step data to free memory ────────────────────────────────
        with self._lock:
            self._steps.clear()
            self._out_hs_sum = None
            self._out_last_hs = None
            self._thinking_hs_buffer = None
            self._finalized = True

    # ------------------------------------------------------------------
    # Finalization helpers
    # ------------------------------------------------------------------

    def _finalize_full_output_mean(self, out_hs: List[np.ndarray]) -> None:
        """Compute output_hidden_states_mean from the full output_hidden_states."""
        out_means = []
        for hs in out_hs:
            if hs.ndim == 3 and hs.shape[1] > 0:
                out_means.append(hs.mean(axis=1))
            else:
                out_means.append(None)
        if any(m is not None for m in out_means):
            sample = next(m for m in out_means if m is not None)
            self.output_hidden_states_mean = np.stack(
                [m if m is not None else np.zeros_like(sample) for m in out_means],
                axis=0,
            )

    def _finalize_last_output(self, out_hs: List[np.ndarray]) -> None:
        """Extract last_output_hidden_state from output_hidden_states."""
        if not out_hs or out_hs[0].shape[1] == 0:
            return
        self.last_output_hidden_state = np.stack(
            [hs[0, -1, :].astype(np.float32) for hs in out_hs], axis=0
        )   # (num_layers, hidden_dim)

    def _finalize_streaming_output(self) -> None:
        """Build output_hidden_states_mean and last_output_hidden_state from
        the streaming state accumulated in _add_step."""
        if self._out_hs_sum is None or self._out_hs_count == 0:
            return
        # Mean: (num_layers, 1, hidden) — keeps batch dim like the full-mode array
        means = [
            (s / self._out_hs_count)[np.newaxis, :].astype(np.float32)
            for s in self._out_hs_sum
        ]
        self.output_hidden_states_mean = np.stack(means, axis=0)
        # Last token: (num_layers, hidden_dim)
        if self._out_last_hs is not None:
            self.last_output_hidden_state = np.stack(
                [v.astype(np.float32) for v in self._out_last_hs], axis=0
            )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def num_layers(self) -> Optional[int]:
        if self.input_hidden_states is not None:
            return len(self.input_hidden_states)
        return None

    @property
    def num_output_tokens(self) -> Optional[int]:
        if self.output_hidden_states is not None:
            if not self.output_hidden_states:
                return 0
            hs = self.output_hidden_states[0]
            return hs.shape[1] if hs.ndim == 3 else 0
        # Streaming mode: output_hidden_states not stored; use step count
        count = self._out_hs_count
        return count if count > 0 else None

    @property
    def is_finalized(self) -> bool:
        return self._finalized

    @property
    def is_encoder_decoder(self) -> bool:
        """True when encoder_hidden_states were captured (T5, BART, …)."""
        return self.encoder_hidden_states is not None

    def __repr__(self) -> str:
        enc = (f", enc_layers={len(self.encoder_hidden_states)}"
               if self.encoder_hidden_states else "")
        return (
            f"InternalsRun("
            f"run_id={self.run_id!r}, "
            f"input_len={self.input_len}, "
            f"num_output_tokens={self.num_output_tokens}, "
            f"num_layers={self.num_layers}"
            f"{enc}, "
            f"finalized={self._finalized})"
        )
