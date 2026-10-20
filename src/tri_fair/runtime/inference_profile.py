"""Set inference controls before vLLM imports; never mislabel a run's profile."""

from __future__ import annotations

import os
import sys


def bootstrap_batch_invariance(argv=None):
    """CLI entry points call this before importing Promptolution or vLLM."""
    argv = sys.argv[1:] if argv is None else argv
    if "--batch-invariant" not in argv:
        return
    if "vllm" in sys.modules and os.environ.get("VLLM_BATCH_INVARIANT") != "1":
        raise RuntimeError(
            "Batch invariance requested after vLLM import. Start a fresh process "
            "with VLLM_BATCH_INVARIANT=1 and --batch-invariant."
        )
    os.environ["VLLM_BATCH_INVARIANT"] = "1"


def require_matching_profile(enabled):
    """Programmatic callers must initialize the environment before imports."""
    actual = os.environ.get("VLLM_BATCH_INVARIANT", "0")
    if actual not in {"0", "1"}:
        raise ValueError("VLLM_BATCH_INVARIANT must be 0 or 1")
    if bool(enabled) != (actual == "1"):
        raise RuntimeError(
            "Inference profile mismatch: --batch-invariant must agree with "
            "VLLM_BATCH_INVARIANT. Set both before starting Python, or disable both."
        )
