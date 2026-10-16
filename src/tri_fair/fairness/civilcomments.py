"""Civil Comments WILDS worst-group accuracy objective."""

from __future__ import annotations
from collections.abc import Sequence
from typing import Any
import pandas as pd
from tri_fair.fairness.base import FairnessMetricResult

DEFAULT_IDENTITIES = (
    "male",
    "female",
    "lgbtq",
    "christian",
    "muslim",
    "other_religions",
    "black",
    "white",
)


def compute_civilcomments_fairness(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    metadata: pd.DataFrame,
    *,
    min_group_count: int,
    min_valid_groups: int = 12,
    identity_columns: Sequence[str] = DEFAULT_IDENTITIES,
    **_: Any,
) -> FairnessMetricResult:
    if len(y_true) != len(y_pred) or len(y_true) != len(metadata):
        raise ValueError("Civil Comments fairness inputs must have equal length")
    work = metadata.copy().reset_index(drop=True)
    work["y_true"] = [str(value).casefold() for value in y_true]
    work["y_pred"] = [str(value).casefold() for value in y_pred]
    work["correct"] = work["y_true"] == work["y_pred"]
    group_accuracy: dict[str, float] = {}
    support: dict[str, int] = {}
    valid_values: list[float] = []
    labels = sorted(work["y_true"].unique())
    for identity in identity_columns:
        if identity not in work.columns:
            raise ValueError(
                f"Civil Comments metadata is missing identity column {identity!r}"
            )
        identity_mask = work[identity].fillna(0).astype(int) == 1
        for label in labels:
            group = work[identity_mask & (work["y_true"] == label)]
            key = f"{identity}/{label}"
            support[key] = int(len(group))
            if len(group) < min_group_count:
                continue
            accuracy = float(group["correct"].mean())
            group_accuracy[key] = accuracy
            valid_values.append(accuracy)
    ready = len(valid_values) >= min_valid_groups
    worst_group_accuracy = float(min(valid_values)) if valid_values else 0.0
    average_accuracy = float(work["correct"].mean()) if len(work) else 0.0
    loss = 1.0 - worst_group_accuracy
    diagnostics = {
        "metric": "civil_worst_group",
        "worst_group_accuracy": worst_group_accuracy,
        "average_accuracy": average_accuracy,
        "average_minus_worst_gap": average_accuracy - worst_group_accuracy,
        "valid_groups": len(valid_values),
        "required_groups": min_valid_groups,
        "group_accuracy": group_accuracy,
    }
    return FairnessMetricResult(
        loss=loss, ready=ready, diagnostics=diagnostics, support=support
    )
