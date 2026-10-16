"""Objective-space transformations, Pareto operations, and frozen normalization bounds."""

from __future__ import annotations
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import pandas as pd

OBJECTIVE_COLUMNS = {
    "dev": ("dev_quality", "dev_cost", "dev_fairness"),
    "test": ("test_quality", "test_cost", "test_fairness"),
}


@dataclass(frozen=True)
class Bounds:
    """Affine bounds for a minimize-all three-objective vector.

    The coordinate order is ``quality_loss, cost, unfairness``.
    """

    minimum: np.ndarray
    maximum: np.ndarray
    source: str = "development_only"

    def __post_init__(self) -> None:
        minimum = np.asarray(self.minimum, dtype=float).reshape(-1)
        maximum = np.asarray(self.maximum, dtype=float).reshape(-1)
        if minimum.shape != (3,) or maximum.shape != (3,):
            raise ValueError("Bounds must contain exactly three objective values")
        if np.any(~np.isfinite(minimum)) or np.any(~np.isfinite(maximum)):
            raise ValueError("Bounds must be finite")
        if np.any(maximum <= minimum):
            raise ValueError(
                f"Each maximum must exceed its minimum: {minimum}, {maximum}"
            )
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    @property
    def span(self) -> np.ndarray:
        return self.maximum - self.minimum

    def normalize(self, values: np.ndarray, *, clip: bool = False) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        normalized = (values - self.minimum) / self.span
        return np.clip(normalized, 0.0, 1.0) if clip else normalized

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        return self.minimum + values * self.span

    def to_json(self) -> dict[str, Any]:
        return {
            "minimum": self.minimum.tolist(),
            "maximum": self.maximum.tolist(),
            "source": self.source,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "Bounds":
        return cls(
            minimum=np.asarray(payload["minimum"], dtype=float),
            maximum=np.asarray(payload["maximum"], dtype=float),
            source=str(payload.get("source", "unknown")),
        )


class BoundsStore:
    """Per-model/per-dataset normalization bounds.

    Bounds should be created once from development data at the first complete
    1M analysis and then reused unchanged for 5M and 7.5M.  This prevents a
    moving normalization scale from fabricating budget improvements.
    """

    VERSION = 1

    def __init__(self, entries: Mapping[tuple[str, str], Bounds] | None = None) -> None:
        self.entries: dict[tuple[str, str], Bounds] = dict(entries or {})

    def __contains__(self, key: tuple[str, str]) -> bool:
        return key in self.entries

    def __getitem__(self, key: tuple[str, str]) -> Bounds:
        try:
            return self.entries[key]
        except KeyError as error:
            known = ", ".join(
                (f"{dataset}/{model}" for dataset, model in sorted(self.entries))
            )
            raise KeyError(
                f"No normalization bounds for {key}; known entries: {known}"
            ) from error

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "coordinate_order": ["quality_loss", "cost", "unfairness"],
            "entries": {
                f"{dataset}/{model}": bounds.to_json()
                for (dataset, model), bounds in sorted(self.entries.items())
            },
        }
        temporary = target.with_suffix(target.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        temporary.replace(target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "BoundsStore":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if int(payload.get("version", -1)) != cls.VERSION:
            raise ValueError(
                f"Unsupported normalization-bounds version {payload.get('version')}; expected {cls.VERSION}"
            )
        entries: dict[tuple[str, str], Bounds] = {}
        for key, value in payload.get("entries", {}).items():
            dataset, model = key.split("/", 1)
            entries[dataset, model] = Bounds.from_json(value)
        return cls(entries)


def objective_matrix(frame: pd.DataFrame, split: str) -> np.ndarray:
    """Return a minimize-all matrix: ``1-quality, cost, unfairness``."""
    if split not in OBJECTIVE_COLUMNS:
        raise ValueError(
            f"Unknown split {split!r}; choose from {tuple(OBJECTIVE_COLUMNS)}"
        )
    quality_col, cost_col, fairness_col = OBJECTIVE_COLUMNS[split]
    missing = {quality_col, cost_col, fairness_col} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing {split} objective columns: {sorted(missing)}")
    quality = pd.to_numeric(frame[quality_col], errors="coerce").to_numpy(dtype=float)
    cost = pd.to_numeric(frame[cost_col], errors="coerce").to_numpy(dtype=float)
    fairness = pd.to_numeric(frame[fairness_col], errors="coerce").to_numpy(dtype=float)
    return np.column_stack([1.0 - quality, cost, fairness])


def valid_objective_rows(
    frame: pd.DataFrame, *, require_test: bool = True
) -> np.ndarray:
    dev = objective_matrix(frame, "dev")
    mask = np.all(np.isfinite(dev), axis=1)
    if require_test:
        test = objective_matrix(frame, "test")
        mask &= np.all(np.isfinite(test), axis=1)
    if "dev_fairness_ready" in frame:
        mask &= frame["dev_fairness_ready"].fillna(False).astype(bool).to_numpy()
    if require_test and "test_fairness_ready" in frame:
        mask &= frame["test_fairness_ready"].fillna(False).astype(bool).to_numpy()
    return mask


def dominates(a: np.ndarray, b: np.ndarray, *, atol: float = 0.0) -> bool:
    """Whether minimize-all vector ``a`` Pareto-dominates ``b``."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return bool(np.all(a <= b + atol) and np.any(a < b - atol))


def pareto_mask(values: np.ndarray, *, atol: float = 0.0) -> np.ndarray:
    """Return a boolean mask identifying non-dominated rows.

    Duplicate vectors are retained because they may correspond to distinct
    prompts.  Call ``deduplicate_vectors`` when a metric should count one copy.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must be a two-dimensional array")
    if len(values) == 0:
        return np.zeros(0, dtype=bool)
    finite = np.all(np.isfinite(values), axis=1)
    keep = finite.copy()
    for index in np.flatnonzero(finite):
        other = values[finite]
        candidate = values[index]
        dominated = np.all(other <= candidate + atol, axis=1) & np.any(
            other < candidate - atol, axis=1
        )
        if np.any(dominated):
            keep[index] = False
    return keep


def non_dominated_front(values: np.ndarray, *, atol: float = 0.0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[pareto_mask(values, atol=atol)]


def deduplicate_vectors(values: np.ndarray, *, decimals: int = 12) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values.reshape(0, values.shape[1] if values.ndim == 2 else 0)
    rounded = np.round(values, decimals=decimals)
    _, unique_indices = np.unique(rounded, axis=0, return_index=True)
    return values[np.sort(unique_indices)]


def crowding_distance(values: np.ndarray) -> np.ndarray:
    """NSGA-II crowding distance for an arbitrary number of objectives."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must be two-dimensional")
    n_points, n_objectives = values.shape
    if n_points <= 2:
        return np.full(n_points, np.inf)
    distance = np.zeros(n_points, dtype=float)
    for objective in range(n_objectives):
        order = np.argsort(values[:, objective], kind="mergesort")
        minimum = values[order[0], objective]
        maximum = values[order[-1], objective]
        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf
        span = maximum - minimum
        if not np.isfinite(span) or span <= 0:
            continue
        previous_values = values[order[:-2], objective]
        next_values = values[order[2:], objective]
        distance[order[1:-1]] += (next_values - previous_values) / span
    return distance


def build_development_bounds(
    evaluations: pd.DataFrame,
    *,
    cost_margin: float = 1.05,
    source: str = "development_only_semantic_quality_and_fairness",
) -> BoundsStore:
    """Build stable bounds without consulting test objectives.

    Quality loss and unfairness use their semantic range ``[0, 1]``.  Cost uses
    ``[0, max_dev_cost * cost_margin]`` for each model/dataset.  Freeze the
    resulting JSON after the 1M stage and reuse it for later budget stages.
    """
    if cost_margin <= 1.0:
        raise ValueError("cost_margin must exceed 1.0")
    entries: dict[tuple[str, str], Bounds] = {}
    for (dataset, model), group in evaluations.groupby(["dataset", "model"], sort=True):
        dev = objective_matrix(group, "dev")
        finite_cost = dev[np.isfinite(dev[:, 1]), 1]
        if not len(finite_cost):
            raise ValueError(f"No finite development costs for {dataset}/{model}")
        maximum_cost = max(float(np.max(finite_cost)) * cost_margin, 1e-12)
        entries[str(dataset), str(model)] = Bounds(
            minimum=np.asarray([0.0, 0.0, 0.0]),
            maximum=np.asarray([1.0, maximum_cost, 1.0]),
            source=source,
        )
    return BoundsStore(entries)


def set_weakly_dominates(a: np.ndarray, b: np.ndarray, *, atol: float = 1e-12) -> bool:
    """Whether every point in set ``b`` is weakly dominated by a point in ``a``."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return False
    for point in b:
        covered = np.any(np.all(a <= point + atol, axis=1))
        if not covered:
            return False
    return True
