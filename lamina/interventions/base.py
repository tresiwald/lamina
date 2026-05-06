"""
Abstract base class for lamina interventions.

An intervention installs PyTorch hooks that *modify* tensors during the
forward pass, as opposed to extractors which only read them.  Interventions
and extractors are fully composable — they are independent hooks registered
on the same model instance and do not interfere with each other.

Design principles
-----------------
* **Context-manager first**: the idiomatic usage is ``with patch(model): ...``
  which guarantees hooks are always removed, even on exceptions.
* **Stackable**: multiple interventions can be active simultaneously; they
  are applied in registration order (same as PyTorch hook semantics).
* **Framework-agnostic interface**: ``install`` / ``remove`` work with any
  object that supports ``register_forward_hook`` and
  ``register_forward_pre_hook`` — not just HuggingFace models.

Planned concrete implementations
---------------------------------
ActivationPatch (lamina.interventions.activation_patch)
    Replace ``output.hidden_states[layer_idx]`` with a pre-stored tensor.
    Implements a single forward-hook that intercepts the model output and
    swaps the hidden state for the chosen layer::

        patch = ActivationPatch(layer=12, value=cached_hs[12][0])  # (seq, h)
        with patch(model):
            run_forward(model, **inputs)

AttentionMask (lamina.interventions.attention_mask)
    Zero out or re-scale specific heads / position pairs in a chosen layer.
    Registers a forward hook on the attention sub-module directly::

        ablate = AttentionMask(layer=8, heads=[0, 1, 4])
        with ablate(model):
            model.generate(input_ids, max_new_tokens=20)

SteeringVector (lamina.interventions.steering)
    Add ``scale * vector`` to the residual stream at a chosen layer.
    Registers a post-layer forward hook that modifies the hidden state
    in-place before it is passed to the next layer::

        steer = SteeringVector(layer=16, vector=direction, scale=20.0)
        with steer(model):
            model.generate(input_ids, max_new_tokens=50)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List


class Intervention(ABC):
    """
    Abstract base for model interventions.

    Subclasses must implement ``install`` and ``remove``; the context-manager
    protocol (``__call__``) is provided for free.
    """

    @abstractmethod
    def install(self, model: Any) -> List[Any]:
        """
        Register hooks on *model* and return a list of hook handles.

        Parameters
        ----------
        model : any object with ``register_forward_hook`` /
                ``register_forward_pre_hook``

        Returns
        -------
        list
            Opaque hook handles that must be passed back to ``remove()``.
        """

    @abstractmethod
    def remove(self, handles: List[Any]) -> None:
        """
        Remove previously installed hooks.

        Parameters
        ----------
        handles : list
            The list returned by the corresponding ``install()`` call.
        """

    def __call__(self, model: Any) -> "_InterventionContext":
        """
        Return a context manager that installs hooks on entry and removes
        them on exit::

            with my_intervention(model):
                model.generate(input_ids, max_new_tokens=20)
        """
        return _InterventionContext(self, model)


class _InterventionContext:
    """Context manager returned by ``Intervention.__call__``."""

    def __init__(self, intervention: Intervention, model: Any) -> None:
        self._intervention = intervention
        self._model = model
        self._handles: List[Any] = []

    def __enter__(self) -> "_InterventionContext":
        self._handles = self._intervention.install(self._model)
        return self

    def __exit__(self, *_: Any) -> None:
        self._intervention.remove(self._handles)
        self._handles = []
