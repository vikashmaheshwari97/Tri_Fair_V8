"""Shared types and helpers for Tri-Fair fairness objectives."""

from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Protocol
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FairnessMetricResult:
    """Result returned by every fairness objective.

    ``loss`` is always an unfairness quantity to minimize.  ``ready`` indicates
    whether the subset contains enough group support for a reliable dominance
    comparison.  A non-ready result may still contain diagnostics, but Tri-Fair
    will not reject a challenger on its basis.
    """

    loss: float
    ready: bool
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    support: Dict[str, int] = field(default_factory=dict)


class FairnessMetric(Protocol):
    def __call__(
        self,
        y_true: Sequence[str],
        y_pred: Sequence[str],
        metadata: pd.DataFrame,
        *,
        min_group_count: int,
        **kwargs: Any,
    ) -> FairnessMetricResult: ...


def safe_rms(values: Sequence[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(array))))


def json_safe(value: Any) -> Any:
    """Recursively convert NumPy/Pandas values into JSON-compatible objects."""
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and (not np.isfinite(value)):
        return None
    return value
