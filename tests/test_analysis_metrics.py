from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from tri_fair.reporting.metrics import (
    approximation_sets,
    hypervolume_2d,
    hypervolume_3d,
    noisy_r2,
    representative_solutions,
)
from tri_fair.reporting.objectives import Bounds, pareto_mask, set_weakly_dominates


class AnalysisMetricTests(unittest.TestCase):
    def test_pareto_mask(self) -> None:
        values = np.asarray(
            [
                [0.2, 0.8, 0.5],
                [0.3, 0.7, 0.4],
                [0.4, 0.9, 0.7],  # dominated by the first two
            ]
        )
        np.testing.assert_array_equal(pareto_mask(values), [True, True, False])

    def test_hypervolume_single_box(self) -> None:
        point = np.asarray([[0.2, 0.3, 0.4]])
        expected = (1.0 - 0.2) * (1.0 - 0.3) * (1.0 - 0.4)
        self.assertAlmostEqual(hypervolume_3d(point, [1.0, 1.0, 1.0]), expected)

    def test_hypervolume_union_two_boxes(self) -> None:
        points = np.asarray([[0.2, 0.8], [0.8, 0.2]])
        # 0.8*0.2 + 0.2*0.8 - overlap 0.2*0.2
        self.assertAlmostEqual(hypervolume_2d(points, [1.0, 1.0]), 0.28)

    def test_approximation_sets(self) -> None:
        values = np.asarray([[0.2, 0.2, 0.2], [0.4, 0.4, 0.4]])
        sets = approximation_sets(values)
        np.testing.assert_array_equal(sets.optimistic_indices, [0])
        np.testing.assert_array_equal(sets.pessimistic_indices, [1])

    def test_noisy_r2_stable(self) -> None:
        dev = np.asarray([[0.1, 0.8, 0.4], [0.8, 0.1, 0.4]])
        self.assertAlmostEqual(
            noisy_r2(dev, dev, n_preferences=100, seed=7),
            noisy_r2(dev, dev, n_preferences=100, seed=7),
        )

    def test_set_dominance(self) -> None:
        a = np.asarray([[0.1, 0.2, 0.3]])
        b = np.asarray([[0.2, 0.3, 0.4], [0.1, 0.4, 0.5]])
        self.assertTrue(set_weakly_dominates(a, b))

    def test_representative_selection(self) -> None:
        frame = pd.DataFrame(
            {
                "prompt": ["quality", "cost", "fair", "balanced"],
                "dev_quality": [0.95, 0.75, 0.80, 0.88],
                "dev_cost": [0.9, 0.1, 0.5, 0.4],
                "dev_fairness": [0.5, 0.5, 0.1, 0.25],
                "test_quality": [0.94, 0.74, 0.79, 0.87],
                "test_cost": [0.9, 0.1, 0.5, 0.4],
                "test_fairness": [0.52, 0.48, 0.12, 0.24],
            }
        )
        bounds = Bounds(np.zeros(3), np.ones(3))
        selected = representative_solutions(frame, bounds)
        self.assertEqual(selected["quality_first"].prompt, "quality")
        self.assertEqual(selected["cost_first"].prompt, "cost")
        self.assertEqual(selected["fairness_first"].prompt, "fair")


if __name__ == "__main__":
    unittest.main()
