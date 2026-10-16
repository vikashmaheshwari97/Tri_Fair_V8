"""Three-objective evaluation metrics for Tri-Fair."""

from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from .objectives import (
    Bounds,
    deduplicate_vectors,
    non_dominated_front,
    objective_matrix,
    pareto_mask,
    valid_objective_rows,
)

DEFAULT_REFERENCE_POINT = np.asarray([1.1, 1.1, 1.1], dtype=float)


@dataclass(frozen=True)
class ApproximationSets:
    optimistic_indices: np.ndarray
    pessimistic_indices: np.ndarray


@dataclass(frozen=True)
class RepresentativeSelection:
    policy: str
    row_position: int
    prompt: str
    dev_quality: float
    dev_cost: float
    dev_fairness: float
    test_quality: float
    test_cost: float
    test_fairness: float

    def to_dict(self, prefix: str | None = None) -> dict[str, Any]:
        values = {
            "policy": self.policy,
            "row_position": self.row_position,
            "prompt": self.prompt,
            "dev_quality": self.dev_quality,
            "dev_cost": self.dev_cost,
            "dev_fairness": self.dev_fairness,
            "test_quality": self.test_quality,
            "test_cost": self.test_cost,
            "test_fairness": self.test_fairness,
        }
        if prefix is None:
            return values
        return {
            f"{prefix}_{key}": value for key, value in values.items() if key != "policy"
        }


def _filter_hv_points(values: np.ndarray, reference_point: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    reference_point = np.asarray(reference_point, dtype=float).reshape(-1)
    if values.ndim != 2 or values.shape[1] != len(reference_point):
        raise ValueError("values and reference_point dimensions do not match")
    finite = values[np.all(np.isfinite(values), axis=1)]
    if not len(finite):
        return finite
    inside = finite[np.all(finite < reference_point, axis=1)]
    if not len(inside):
        return inside
    return deduplicate_vectors(non_dominated_front(inside))


def hypervolume_2d(values: np.ndarray, reference_point: Sequence[float]) -> float:
    """Exact union area of minimize-all 2-D boxes against a reference point."""
    reference = np.asarray(reference_point, dtype=float).reshape(2)
    points = _filter_hv_points(np.asarray(values, dtype=float), reference)
    if not len(points):
        return 0.0
    order = np.argsort(points[:, 0], kind="mergesort")
    points = points[order]
    x_values = np.unique(points[:, 0])
    boundaries = np.concatenate([x_values, [reference[0]]])
    area = 0.0
    best_y = reference[1]
    cursor = 0
    for index, x_value in enumerate(x_values):
        while cursor < len(points) and points[cursor, 0] <= x_value:
            best_y = min(best_y, points[cursor, 1])
            cursor += 1
        width = boundaries[index + 1] - x_value
        height = reference[1] - best_y
        if width > 0 and height > 0:
            area += width * height
    return float(area)


def hypervolume_3d(
    values: np.ndarray, reference_point: Sequence[float] = DEFAULT_REFERENCE_POINT
) -> float:
    """Exact 3-D hypervolume for minimize-all objective vectors.

    The implementation sweeps the first objective and computes the exact 2-D
    union area in every slab.  It avoids a mandatory ``pymoo`` dependency and is
    sufficient for the three objectives used by Tri-Fair.
    """
    reference = np.asarray(reference_point, dtype=float).reshape(3)
    points = _filter_hv_points(np.asarray(values, dtype=float), reference)
    if not len(points):
        return 0.0
    order = np.argsort(points[:, 0], kind="mergesort")
    points = points[order]
    x_values = np.unique(points[:, 0])
    boundaries = np.concatenate([x_values, [reference[0]]])
    volume = 0.0
    for index, x_value in enumerate(x_values):
        active = points[points[:, 0] <= x_value][:, 1:]
        width = boundaries[index + 1] - x_value
        if width <= 0:
            continue
        area = hypervolume_2d(active, reference[1:])
        volume += width * area
    return float(volume)


def weighted_tchebycheff(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if values.ndim != 2 or values.shape[1] != len(weights):
        raise ValueError("values and weights dimensions do not match")
    return np.max(values * weights[None, :], axis=1)


def noisy_r2(
    dev_values: np.ndarray,
    test_values: np.ndarray,
    *,
    n_preferences: int = 1000,
    seed: int = 2026,
) -> float:
    """Noisy R2 using development selection and holdout utility evaluation."""
    dev_values = np.asarray(dev_values, dtype=float)
    test_values = np.asarray(test_values, dtype=float)
    if dev_values.shape != test_values.shape:
        raise ValueError(
            "Development and test objective matrices must have equal shape"
        )
    if len(dev_values) == 0:
        return float("nan")
    mask = pareto_mask(dev_values)
    dev_front = dev_values[mask]
    test_corresponding = test_values[mask]
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(dev_values.shape[1]), size=int(n_preferences))
    holdout_utilities = np.empty(len(weights), dtype=float)
    for index, weight in enumerate(weights):
        development_utility = weighted_tchebycheff(dev_front, weight)
        selected = int(np.argmin(development_utility))
        holdout_utilities[index] = weighted_tchebycheff(
            test_corresponding[selected : selected + 1], weight
        )[0]
    return float(np.mean(holdout_utilities))


def approximation_sets(test_values: np.ndarray) -> ApproximationSets:
    """Construct optimistic and pessimistic holdout sets.

    * Optimistic: solutions that remain non-dominated on holdout data.
    * Pessimistic: solutions that dominate no other development-selected
      solution on holdout data.

    For a perfectly stable non-dominated set the two sets coincide.  When a
    development-selected solution becomes dominated on holdout data, it moves
    to the pessimistic boundary and widens the approximation gap.
    """
    values = np.asarray(test_values, dtype=float)
    if values.ndim != 2:
        raise ValueError("test_values must be two-dimensional")
    if len(values) == 0:
        return ApproximationSets(np.asarray([], dtype=int), np.asarray([], dtype=int))
    dominates_matrix = np.all(
        values[:, None, :] <= values[None, :, :], axis=2
    ) & np.any(values[:, None, :] < values[None, :, :], axis=2)
    dominated_by_any = np.any(dominates_matrix, axis=0)
    dominates_any = np.any(dominates_matrix, axis=1)
    return ApproximationSets(
        optimistic_indices=np.flatnonzero(~dominated_by_any),
        pessimistic_indices=np.flatnonzero(~dominates_any),
    )


def optimistic_pessimistic_hypervolume(
    test_values: np.ndarray, reference_point: Sequence[float] = DEFAULT_REFERENCE_POINT
) -> tuple[float, float, float, ApproximationSets]:
    sets = approximation_sets(test_values)
    optimistic = hypervolume_3d(test_values[sets.optimistic_indices], reference_point)
    pessimistic = hypervolume_3d(test_values[sets.pessimistic_indices], reference_point)
    gap = max(0.0, optimistic - pessimistic)
    return (optimistic, pessimistic, gap, sets)


def _row_value(row: pd.Series, column: str) -> float:
    value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
    return float(value)


def _selection_from_position(
    frame: pd.DataFrame, position: int, policy: str
) -> RepresentativeSelection:
    row = frame.iloc[int(position)]
    return RepresentativeSelection(
        policy=policy,
        row_position=int(position),
        prompt=str(row.get("prompt", "")),
        dev_quality=_row_value(row, "dev_quality"),
        dev_cost=_row_value(row, "dev_cost"),
        dev_fairness=_row_value(row, "dev_fairness"),
        test_quality=_row_value(row, "test_quality"),
        test_cost=_row_value(row, "test_cost"),
        test_fairness=_row_value(row, "test_fairness"),
    )


def representative_solutions(
    frame: pd.DataFrame, bounds: Bounds
) -> dict[str, RepresentativeSelection]:
    """Select interpretable prompts using development objectives only."""
    if frame.empty:
        return {}
    dev = objective_matrix(frame, "dev")
    front_mask = pareto_mask(dev)
    front_positions = np.flatnonzero(front_mask)
    front = frame.iloc[front_positions].reset_index(drop=True)
    front_dev = objective_matrix(front, "dev")
    normalized = bounds.normalize(front_dev, clip=False)
    selected_local: dict[str, int] = {
        "quality_first": int(np.argmin(front_dev[:, 0])),
        "cost_first": int(np.argmin(front_dev[:, 1])),
        "fairness_first": int(np.argmin(front_dev[:, 2])),
        "balanced": int(np.argmin(np.linalg.norm(normalized, axis=1))),
    }
    return {
        policy: _selection_from_position(front, position, policy)
        for policy, position in selected_local.items()
    }


def analyse_evaluated_run(
    frame: pd.DataFrame,
    bounds: Bounds,
    *,
    n_preferences: int = 1000,
    preference_seed: int = 2026,
    reference_point: Sequence[float] = DEFAULT_REFERENCE_POINT,
) -> dict[str, Any]:
    """Compute all metrics for one run at one evaluated checkpoint."""
    mask = valid_objective_rows(frame, require_test=True)
    valid = frame.loc[mask].reset_index(drop=True)
    if valid.empty:
        raise ValueError(
            "No rows have finite, fairness-ready development and test objectives"
        )
    dev_raw = objective_matrix(valid, "dev")
    test_raw = objective_matrix(valid, "test")
    dev = bounds.normalize(dev_raw, clip=False)
    test = bounds.normalize(test_raw, clip=False)
    dev_front_mask = pareto_mask(dev)
    dev_front = dev[dev_front_mask]
    test_selected = test[dev_front_mask]
    raw_front = valid.loc[dev_front_mask].reset_index(drop=True)
    hv_dev = hypervolume_3d(dev_front, reference_point)
    hv_opt, hv_pes, hv_gap, sets = optimistic_pessimistic_hypervolume(
        test_selected, reference_point
    )
    nr2 = noisy_r2(dev, test, n_preferences=n_preferences, seed=preference_seed)
    representatives = representative_solutions(valid, bounds)
    fairness_delta = pd.to_numeric(
        raw_front["test_fairness"], errors="coerce"
    ).to_numpy(dtype=float) - pd.to_numeric(
        raw_front["dev_fairness"], errors="coerce"
    ).to_numpy(dtype=float)
    result: dict[str, Any] = {
        "n_candidates": int(len(valid)),
        "n_dev_pareto": int(dev_front_mask.sum()),
        "n_test_optimistic": int(len(sets.optimistic_indices)),
        "n_test_pessimistic": int(len(sets.pessimistic_indices)),
        "hv_dev_3d": hv_dev,
        "hv_test_optimistic_3d": hv_opt,
        "hv_test_pessimistic_3d": hv_pes,
        "approximation_gap_3d": hv_gap,
        "noisy_r2_3d": nr2,
        "mean_dev_front_test_quality": float(raw_front["test_quality"].mean()),
        "mean_dev_front_test_cost": float(raw_front["test_cost"].mean()),
        "mean_dev_front_test_fairness": float(raw_front["test_fairness"].mean()),
        "fairness_generalization_gap": float(np.mean(fairness_delta)),
        "fairness_generalization_gap_abs": float(np.mean(np.abs(fairness_delta))),
        "fairness_generalization_gap_max_abs": float(np.max(np.abs(fairness_delta))),
    }
    for policy, selection in representatives.items():
        result.update(selection.to_dict(prefix=policy))
    return result


def aggregate_run_metrics(run_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate independent seeds while retaining mean, std, and count."""
    identifiers = ["dataset", "model", "optimizer", "budget_checkpoint"]
    excluded = set(identifiers) | {
        "run_key",
        "run_dir",
        "seed",
        "chosen_step",
        "actual_budget_tokens",
    }
    numeric_columns = [
        column
        for column in run_metrics.select_dtypes(include=[np.number]).columns
        if column not in excluded
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in run_metrics.groupby(identifiers, sort=True, dropna=False):
        row = dict(zip(identifiers, keys))
        row["n_runs"] = int(group["seed"].nunique())
        row["actual_budget_tokens_mean"] = float(group["actual_budget_tokens"].mean())
        row["actual_budget_tokens_max"] = int(group["actual_budget_tokens"].max())
        for column in numeric_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = (
                float(values.std(ddof=1)) if values.notna().sum() > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)
