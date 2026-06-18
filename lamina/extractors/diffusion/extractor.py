"""
DiffusionExtractor — stub for discrete / masked diffusion language models.

Discrete diffusion LMs (MDLM, LLaDA, SEDD-style) generate text via
iterative **unmasking** rather than autoregressive token sampling.  Each
denoising step is an independent full-sequence forward pass identical in
shape to a masked-LM forward call (BertForMaskedLM, …).

Current status: **stub**.  The intended API is documented below but the
implementation is not yet complete.

For now, use :func:`lamina.run_diffusion` directly::

    from lamina import run_diffusion

    # Build per-step inputs (partially masked at each noise level)
    step_inputs = [
        {"input_ids": masked_ids_t}
        for masked_ids_t in my_diffusion_schedule(input_ids, n_steps=20)
    ]

    run_ids = run_diffusion(model, step_inputs)
    # One InternalsRun per step — retrieve after finalization:
    runs = [lamina.get_run(rid) for rid in run_ids]

Intended future API
-------------------
Once implemented, ``DiffusionExtractor`` will wrap the denoising loop in a
stateful interface::

    extractor = DiffusionExtractor(model, tokenizer)
    runs = extractor.run_denoising(
        prompt_ids,
        n_steps=20,
        schedule_fn=my_noise_schedule,    # (ids, t) → masked_ids
        unmask_fn=my_unmask_policy,       # (logits, masked_ids, t) → new_ids
    )
    # runs: List[InternalsRun], one per denoising step

Architecture notes
------------------
* Forward pass: ``model(input_ids)`` → ``MaskedLMOutput`` with
  ``logits`` of shape ``(batch, seq, vocab)`` — **full sequence**, not just
  the last token.  ``logit_mode`` will be ``"full"`` in lamina's hook.
* No ``model.generate()`` — generation is a custom loop.
* ``output_hidden_states`` at step ``t`` are the representations of the
  *current* partially-unmasked sequence, useful for probing what
  information the model uses at each noise level.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional


class DiffusionExtractor:
    """
    Stateful wrapper for iterative denoising over a masked diffusion LM.

    .. note::
        **Not yet implemented.**  Instantiating this class raises
        :exc:`NotImplementedError`.  Use :func:`lamina.run_diffusion` for
        now; see module docstring for details.

    Parameters (future)
    ----------
    model : PreTrainedModel
        A masked-LM style model (e.g. ``BertForMaskedLM`` or a custom
        diffusion backbone).
    tokenizer : PreTrainedTokenizer
    store : InternalsStore | None
        Lamina store to write runs into.  Defaults to the global singleton.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        store: Optional[Any] = None,
    ) -> None:
        raise NotImplementedError(
            "DiffusionExtractor is not yet implemented.\n"
            "Use lamina.run_diffusion(model, step_inputs) directly:\n\n"
            "    from lamina import run_diffusion\n"
            "    run_ids = run_diffusion(model, step_inputs)\n"
        )

    def run_denoising(
        self,
        prompt_ids: Any,
        n_steps: int,
        schedule_fn: Callable[[Any, int], Any],
        unmask_fn: Callable[[Any, Any, int], Any],
    ) -> List[Any]:
        """
        Run the full denoising chain and return one ``InternalsRun`` per step.

        .. note::
            **Not yet implemented.**

        Parameters
        ----------
        prompt_ids : Tensor | np.ndarray
            The initial (fully masked or noised) token IDs.
        n_steps : int
            Number of denoising steps.
        schedule_fn : callable(ids, step_idx) → noised_ids
            Adds noise at the current noise level.
        unmask_fn : callable(logits, masked_ids, step_idx) → new_ids
            Selects which tokens to unmask given the model's predictions.

        Returns
        -------
        list[InternalsRun]
            One run per denoising step, in order from noisiest to cleanest.
        """
        raise NotImplementedError
