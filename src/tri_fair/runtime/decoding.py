"""Sampling controls for reproducible downstream evaluation and creative meta-search."""

from __future__ import annotations
from typing import Any


def _sampling_params(llm: Any) -> Any:
    params = getattr(llm, "sampling_params", None)
    if params is None:
        raise TypeError("The configured LLM wrapper has no sampling_params attribute")
    return params


def configure_downstream_greedy(llm: Any, *, seed: int) -> Any:
    """Use greedy classification/evaluation decoding to reduce sampling variance.

    Greedy decoding is used only for the downstream classifier and held-out
    evaluator.  The meta-model remains stochastic so it can generate diverse
    prompt mutations. GPU numerical differences may remain across batch shapes.
    """
    params = _sampling_params(llm)
    params.temperature = 0.0
    params.top_p = 1.0
    params.seed = int(seed)
    if hasattr(params, "n"):
        params.n = 1
    if hasattr(params, "best_of"):
        try:
            params.best_of = 1
        except (AttributeError, ValueError):
            pass
    return llm


def configure_meta_search(
    llm: Any, *, seed: int, temperature: float = 0.55, top_p: float = 0.95
) -> Any:
    """Configure moderate stochasticity for crossover and mutation generation."""
    if not 0.0 < float(temperature) <= 2.0:
        raise ValueError("meta temperature must lie in (0, 2]")
    if not 0.0 < float(top_p) <= 1.0:
        raise ValueError("meta top_p must lie in (0, 1]")
    params = _sampling_params(llm)
    params.temperature = float(temperature)
    params.top_p = float(top_p)
    params.seed = int(seed)
    if hasattr(params, "n"):
        params.n = 1
    return llm


def configure_structured_labels(llm: Any, classes) -> Any:
    """Optional vLLM choice grammar preserving the predictor's tagged contract.

    Requires a supported vLLM backend; verify on the selected checkpoint/GPU.
    """
    from vllm.sampling_params import StructuredOutputsParams

    choices = [f"<final_answer>{label}</final_answer>" for label in classes]
    if not choices:
        raise ValueError("classes cannot be empty")
    _sampling_params(llm).structured_outputs = StructuredOutputsParams(choice=choices)
    return llm
