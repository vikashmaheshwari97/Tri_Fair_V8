"""Collection of utility functions."""

from __future__ import annotations
import copy
import hashlib
import importlib
import os
import random
import string
from typing import Any
import numpy as np


def generate_hash_from_string(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generate_random_hash() -> str:
    random_string = "".join(random.choices(string.ascii_letters + string.digits, k=32))
    return generate_hash_from_string(random_string)


def _load_torch_optional():
    try:
        return importlib.import_module("torch")
    except (ImportError, OSError):
        return None


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy and—when available—PyTorch.

    A broken or absent local torch install must not prevent dataset preparation,
    analysis, unit tests or CLI argument inspection.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch = _load_torch_optional()
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def copy_llm(model_obj: Any, llm_attr_name: str = "llm") -> Any:
    """Copy an LLM wrapper while sharing its underlying inference engine."""
    new_obj = type(model_obj).__new__(type(model_obj))
    for attr_name, value in model_obj.__dict__.items():
        if attr_name == llm_attr_name:
            setattr(new_obj, attr_name, value)
        else:
            setattr(new_obj, attr_name, copy.deepcopy(value))
    return new_obj
