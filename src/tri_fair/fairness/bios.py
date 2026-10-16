"""Bias-in-Bios profession-wise gender TPR-gap objective."""

from __future__ import annotations
from collections.abc import Sequence
from typing import Any
import numpy as np
import pandas as pd
from tri_fair.fairness.base import FairnessMetricResult, safe_rms


def compute_bios_fairness(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    metadata: pd.DataFrame,
    *,
    min_group_count: int,
    min_valid_professions: int = 20,
    male_value: int = 0,
    female_value: int = 1,
    **_: Any,
) -> FairnessMetricResult:
    if len(y_true) != len(y_pred) or len(y_true) != len(metadata):
        raise ValueError("Bias-in-Bios fairness inputs must have equal length")
    if "gender" not in metadata.columns:
        raise ValueError("Bias-in-Bios metadata requires a 'gender' column")
    work = metadata.copy().reset_index(drop=True)
    work["y_true"] = [str(value).casefold() for value in y_true]
    work["y_pred"] = [str(value).casefold() for value in y_pred]
    gaps: dict[str, float] = {}
    tprs: dict[str, dict[str, float]] = {}
    support: dict[str, int] = {}
    for profession in sorted(work["y_true"].unique()):
        profession_rows = work[work["y_true"] == profession]
        male_rows = profession_rows[profession_rows["gender"].astype(int) == male_value]
        female_rows = profession_rows[
            profession_rows["gender"].astype(int) == female_value
        ]
        support[f"{profession}/male"] = int(len(male_rows))
        support[f"{profession}/female"] = int(len(female_rows))
        if len(male_rows) < min_group_count or len(female_rows) < min_group_count:
            continue
        male_tpr = float(np.mean(male_rows["y_pred"] == profession))
        female_tpr = float(np.mean(female_rows["y_pred"] == profession))
        gap = male_tpr - female_tpr
        gaps[profession] = gap
        tprs[profession] = {"male": male_tpr, "female": female_tpr}
    values = list(gaps.values())
    loss = safe_rms(values)
    ready = len(values) >= min_valid_professions and np.isfinite(loss)
    if not np.isfinite(loss):
        loss = 1.0
    diagnostics = {
        "metric": "bios_tpr_gap",
        "rms_tpr_gap": loss,
        "max_abs_tpr_gap": float(max((abs(value) for value in values), default=1.0)),
        "mean_signed_tpr_gap": float(np.mean(values)) if values else None,
        "valid_professions": len(values),
        "required_professions": min_valid_professions,
        "tpr_gap_by_profession": gaps,
        "tpr_by_profession_and_gender": tprs,
    }
    return FairnessMetricResult(
        loss=loss, ready=ready, diagnostics=diagnostics, support=support
    )
