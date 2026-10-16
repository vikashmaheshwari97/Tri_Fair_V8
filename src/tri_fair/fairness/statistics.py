"""Statistical helpers for group-fairness objectives.

The functions in this module deliberately avoid SciPy so the metric layer remains
usable in CPU-only smoke tests and on the Rocket login nodes.
"""

from __future__ import annotations
from collections.abc import Sequence
from math import sqrt
from statistics import NormalDist
from typing import Iterable
import numpy as np


def normal_quantile(confidence: float = 0.95) -> float:
    """Return the two-sided standard-normal quantile for ``confidence``."""
    value = float(confidence)
    if not 0.0 < value < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")
    return float(NormalDist().inv_cdf(0.5 + value / 2.0))


def wilson_interval(
    successes: int, total: int, *, confidence: float = 0.95
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    For ``total == 0`` the complete probability range is returned.  That makes
    missing evidence maximally uncertain rather than spuriously fair.
    """
    n = int(total)
    k = int(successes)
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"invalid binomial counts: successes={k}, total={n}")
    if n == 0:
        return (0.0, 1.0)
    z = normal_quantile(confidence)
    phat = k / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    centre = (phat + z2 / (2.0 * n)) / denominator
    radius = z * sqrt(phat * (1.0 - phat) / n + z2 / (4.0 * n * n)) / denominator
    return (max(0.0, centre - radius), min(1.0, centre + radius))


def wilson_width(successes: int, total: int, *, confidence: float = 0.95) -> float:
    lower, upper = wilson_interval(successes, total, confidence=confidence)
    return float(upper - lower)


def smoothed_rate(successes: int, total: int, *, alpha: float = 0.5) -> float:
    """Symmetric Beta-prior posterior mean for a Bernoulli rate."""
    n = int(total)
    k = int(successes)
    prior = float(alpha)
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"invalid binomial counts: successes={k}, total={n}")
    if prior < 0:
        raise ValueError("alpha must be non-negative")
    denominator = n + 2.0 * prior
    if denominator == 0:
        return 0.5
    return float((k + prior) / denominator)


def finite_max(values: Iterable[float], *, default: float = float("inf")) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(np.max(array)) if array.size else float(default)


def readiness_reasons(
    *,
    valid_units: int,
    required_units: int,
    interval_widths: Sequence[float],
    maximum_width: float,
    extra_failures: Sequence[str] = (),
) -> list[str]:
    """Build human-readable reasons why a fairness estimate is not ready."""
    reasons: list[str] = []
    if int(valid_units) < int(required_units):
        reasons.append(
            f"valid units {int(valid_units)} < required {int(required_units)}"
        )
    max_width = finite_max(interval_widths)
    if not np.isfinite(max_width) or max_width > float(maximum_width):
        reasons.append(
            f"maximum confidence-interval width {max_width:.4f} > allowed {float(maximum_width):.4f}"
        )
    reasons.extend((str(value) for value in extra_failures if str(value)))
    return reasons
