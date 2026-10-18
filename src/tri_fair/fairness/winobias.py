"""Paired WinoBias accuracy-gap adaptation with statistical support gates."""

import numpy as np
from tri_fair.fairness.base import FairnessMetricResult
from tri_fair.fairness.statistics import wilson_width


def compute_winobias_fairness(
    y_true,
    y_pred,
    metadata,
    *,
    min_group_count,
    confidence=0.95,
    max_rate_ci_width=0.6,
    min_accuracy=0.5,
    min_class_recall=0.5,
    **kwargs,
):
    if len(y_true) != len(y_pred) or len(y_true) != len(metadata):
        raise ValueError("WinoBias inputs must have equal length")
    work = metadata.copy().reset_index(drop=True)
    work["prediction"] = [str(v).casefold() for v in y_pred]
    work["correct"] = np.array(y_true) == np.array(y_pred)
    support, rates, widths = {}, {}, {}
    for task_type in ["type1", "type2"]:
        for condition in ["pro", "anti"]:
            key = f"{task_type}/{condition}"
            cell = work[(work.task_type == task_type) & (work.stereotype == condition)]
            support[key] = len(cell)
            rates[key] = float(cell.correct.mean()) if len(cell) else float("nan")
            widths[key] = (
                wilson_width(int(cell.correct.sum()), len(cell), confidence=confidence)
                if len(cell)
                else 1.0
            )
    paired = all(
        set(g.stereotype) == {"pro", "anti"} and len(g) == 2
        for _, g in work.groupby("pair_id")
    )
    ready = bool(
        paired
        and all(n >= min_group_count for n in support.values())
        and max(widths.values()) <= max_rate_ci_width
    )
    gaps = {t: abs(rates[f"{t}/pro"] - rates[f"{t}/anti"]) for t in ["type1", "type2"]}
    gap = float(np.sqrt(np.mean(np.square(list(gaps.values())))))
    accuracy = float(work.correct.mean()) if len(work) else 0.0
    # Collapse guard is explicit and identical for every optimizer. Raw gaps
    # remain separately visible rather than presenting chance accuracy as fair.
    penalty = max(0.0, (min_accuracy - accuracy) / max(min_accuracy, 1e-12))
    recalls = {}
    truth = np.array(y_true)
    for label in ["a", "b"]:
        selected = truth == label
        recalls[label] = (
            float(work.loc[selected, "correct"].mean()) if selected.any() else 0.0
        )
    recall_penalty = max(
        0.0, (min_class_recall - min(recalls.values())) / max(min_class_recall, 1e-12)
    )
    penalty = max(penalty, recall_penalty)
    consistency = [
        len(set(g.prediction)) == 1 for _, g in work.groupby("pair_id") if len(g) == 2
    ]
    return FairnessMetricResult(
        loss=max(gap, penalty) if np.isfinite(gap) else 1.0,
        ready=ready,
        diagnostics=dict(
            raw_gap_rms=gap,
            gap_by_type=gaps,
            accuracy_by_cell=rates,
            accuracy=accuracy,
            accuracy_floor_penalty=penalty,
            recall_by_label=recalls,
            min_class_recall=min_class_recall,
            paired_consistency=float(np.mean(consistency)) if consistency else None,
            uncertainty_by_group=widths,
            max_ci_width=max(widths.values()),
            scorer="two-choice accuracy adaptation; not official CoNLL coreference F1",
        ),
        support=support,
    )
