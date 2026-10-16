"""Deterministic selection primitives for maximize-all three-objective search."""

from __future__ import annotations
import numpy as np


def nondominated(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("expected a two-dimensional objective matrix")
    finite = np.isfinite(values).all(axis=1)
    keep = finite.copy()
    for i in np.flatnonzero(finite):
        keep[i] = not np.any(
            np.all(values[finite] >= values[i], axis=1)
            & np.any(values[finite] > values[i], axis=1)
        )
    return keep


def hypervolume(values: np.ndarray, reference=(1.1, 1.1, 1.1)) -> float:
    """Exact minimization HV using pymoo/moocore; no clipped or fabricated points."""
    from pymoo.indicators.hv import HV

    values = np.asarray(values, dtype=float).reshape(-1, 3)
    reference = np.asarray(reference, dtype=float)
    points = values[
        np.isfinite(values).all(axis=1) & np.all(values <= reference, axis=1)
    ]
    return float(HV(ref_point=reference).do(points)) if len(points) else 0.0


def truncate_archive(
    normalised: np.ndarray, capacity: int, directions: np.ndarray
) -> list[int]:
    """Preserve extremes, cover niches, then greedily add marginal-HV winners.

    Every stage has a bounded slot allocation, so reference selection cannot
    silently consume the hypervolume stage's slots. Input scores maximize all
    objectives and use common fixed normalization.
    """
    values = np.asarray(normalised, dtype=float)
    if (
        capacity < 1
        or values.ndim != 2
        or values.shape[1] != 3
        or (not np.isfinite(values).all())
    ):
        raise ValueError("finite n-by-3 values and positive capacity are required")
    if len(values) <= capacity:
        return list(range(len(values)))
    selected: list[int] = []
    for j in range(3):
        i = int(np.argmax(values[:, j]))
        if i not in selected and len(selected) < capacity:
            selected.append(i)
    niche_end = min(
        capacity - 1, len(selected) + max(0, (capacity - len(selected)) // 2)
    )
    losses = 1.0 - values
    for direction in np.asarray(directions):
        if len(selected) >= niche_end:
            break
        achievement = np.max(losses / np.maximum(direction, 0.0001), axis=1)
        for i in np.argsort(achievement, kind="stable"):
            if int(i) not in selected:
                selected.append(int(i))
                break
    while len(selected) < capacity:
        current = hypervolume(losses[selected])
        remaining = [i for i in range(len(values)) if i not in selected]

        def gain(i):
            novelty = min(
                (float(np.linalg.norm(values[i] - values[j])) for j in selected)
            )
            return (hypervolume(losses[selected + [i]]) - current, novelty, -i)

        selected.append(max(remaining, key=gain))
    return selected


def spea2_fitness(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SPEA2 strength/raw fitness plus k-nearest-neighbour density."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("finite objective matrix required")
    dominates = np.all(values[:, None, :] >= values[None, :, :], axis=2) & np.any(
        values[:, None, :] > values[None, :, :], axis=2
    )
    strength = dominates.sum(axis=1)
    raw = dominates.T @ strength
    span = np.ptp(values, axis=0)
    scaled = (values - values.min(axis=0)) / np.where(span > 0, span, 1)
    distance = np.linalg.norm(scaled[:, None] - scaled[None, :], axis=2)
    np.fill_diagonal(distance, np.inf)
    if len(values) < 2:
        density = np.zeros(len(values))
    else:
        k = min(max(1, int(np.sqrt(len(values)))), len(values) - 1)
        density = 1.0 / (np.sort(distance, axis=1)[:, k - 1] + 2.0)
    return (raw + density, distance)


def spea2_select(values: np.ndarray, capacity: int) -> list[int]:
    fitness, distances = spea2_fitness(values)
    selected = np.flatnonzero(fitness < 1.0).tolist()
    if len(selected) < capacity:
        selected = sorted(range(len(values)), key=lambda i: (fitness[i], i))[:capacity]
    while len(selected) > capacity:
        victim = min(
            selected,
            key=lambda i: (
                tuple(sorted((float(distances[i, j]) for j in selected if i != j)))
                + (i,)
            ),
        )
        selected.remove(victim)
    return selected


def simplex_reference_directions(partitions: int) -> np.ndarray:
    if partitions <= 0:
        raise ValueError("reference_partitions must be positive")
    directions: list[tuple[float, float, float]] = []
    for first in range(partitions + 1):
        for second in range(partitions + 1 - first):
            third = partitions - first - second
            directions.append(
                (first / partitions, second / partitions, third / partitions)
            )
    return np.asarray(directions, dtype=float)
