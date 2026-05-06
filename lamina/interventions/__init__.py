"""
lamina.interventions
====================
Active model modifications — hooks that *change* model computations rather
than merely observing them.

Interventions compose freely with extractors: both use
``register_forward_hook`` / ``register_forward_pre_hook`` on the same model,
so you can extract internals from an intervened-upon forward pass::

    import lamina
    from lamina.interventions import ActivationPatch

    # Replace layer 12's output with a pre-cached activation
    patch = ActivationPatch(layer=12, value=source_run.input_hidden_states[12])
    with patch(model):
        lamina.run_forward(model, **inputs)
        run = lamina.get_latest()   # internals of the patched forward pass

Available interventions
-----------------------
ActivationPatch
    Replace a specific layer's hidden-state output with a stored tensor.
    Useful for causal tracing (what happens to the output when layer N sees
    a different activation?).

AttentionMask
    Zero out or scale specific attention heads or position pairs within a
    chosen layer.  Useful for ablation studies ("what does head 4 in layer
    8 contribute?").

SteeringVector
    Add a scaled vector to the residual stream at a chosen layer.  Used for
    representation engineering and activation steering experiments.

Status
------
All three are **conceptually defined** here (documented ABCs and planned
concrete classes).  Implementations will be added in a future release.
"""
from .base import Intervention

__all__ = ["Intervention"]
