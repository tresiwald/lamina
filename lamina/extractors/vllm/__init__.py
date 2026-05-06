"""
lamina.extractors.vllm
======================
vLLM extractor — **not yet implemented**.

This sub-package is a placeholder for a future extractor that hooks into
the vLLM inference engine (``vllm.LLM`` / ``AsyncLLMEngine``) to capture
hidden states and attention weights with the same interface as the HF
extractor.

Planned approach
----------------
vLLM exposes model internals via ``--return-hidden-states`` and the
``RequestOutput`` object.  The extractor will wrap ``LLM.generate()`` and
collect per-step data into the shared ``lamina.core`` ring buffer, keeping
the API identical to the HF path::

    import lamina
    from vllm import LLM

    llm = LLM("meta-llama/Llama-3-8B")
    lamina.extractors.vllm.attach(llm)

    outputs = llm.generate(["Hello, world!"])
    run = lamina.get_latest()

Install with::

    pip install lamina[vllm]
"""

__all__: list = []
