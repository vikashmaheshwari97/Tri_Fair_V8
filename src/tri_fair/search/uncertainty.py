"""Reproducible paired resampling of cached predictions, with no LLM calls.

These are bootstrap diagnostics, not anytime-valid confidence sequences.
"""

from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd


def resample_indices(
    frame: pd.DataFrame, *, metric: str, seed: int, repeats: int
) -> list[np.ndarray]:
    if repeats < 2 or frame.empty:
        raise ValueError("at least two replicates and nonempty evidence are required")
    ids = frame["_review_id"].astype(str).tolist()
    fingerprint = hashlib.sha256((str(seed) + "|" + "|".join(ids)).encode()).digest()
    rng = np.random.default_rng(int.from_bytes(fingerprint[:8], "little"))
    if metric.startswith("bbq") and "template_id" in frame:
        strata = {}
        for (category, _), positions in frame.groupby(
            ["category", "template_id"], sort=True, dropna=False
        ).indices.items():
            strata.setdefault(category, []).append(np.asarray(positions, dtype=int))
        return [
            np.concatenate(
                [
                    units[i]
                    for units in strata.values()
                    for i in rng.integers(0, len(units), len(units))
                ]
            )
            for _ in range(repeats)
        ]
    cols = (
        ["_review_target", "gender"]
        if metric.startswith("bios")
        else ["_review_target"]
    )
    strata = [
        np.asarray(v, dtype=int)
        for v in frame.groupby(cols, sort=True, dropna=False).indices.values()
    ]
    return [
        np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
        for _ in range(repeats)
    ]


def pooled_bootstrap(
    task, prompts, blocks, *, cost_in: float, cost_out: float, seed: int, repeats: int
):
    rows = [task.cached_prediction_frame(p, blocks) for p in prompts]
    if not rows:
        return np.empty((repeats, 0, 3))
    identifiers = rows[0]["_review_id"].tolist()
    if any((r["_review_id"].tolist() != identifiers for r in rows)):
        raise ValueError("paired bootstrap requires identical example IDs and order")
    indices = resample_indices(
        rows[0], metric=task.fairness_metric_name, seed=seed, repeats=repeats
    )
    return np.asarray(
        [
            [task.objectives_from_frame(r.iloc[idx], cost_in, cost_out) for r in rows]
            for idx in indices
        ]
    )
