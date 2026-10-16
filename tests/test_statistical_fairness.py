import numpy as np
import pandas as pd

from tri_fair.fairness.statistical_metrics import (
    compute_bbq_statistical_fairness,
    compute_bios_statistical_tpr_gap,
    compute_civil_equalized_odds,
)
from tri_fair.fairness.statistics import wilson_width


def test_wilson_width_shrinks_with_support():
    assert wilson_width(1, 2) > wilson_width(10, 20) > wilson_width(50, 100)


def test_bbq_unknown_collapse_is_not_fair():
    metadata = pd.DataFrame(
        {
            "category": ["Age"] * 8,
            "context_condition": ["ambig"] * 4 + ["disambig"] * 4,
            "target_loc": [0, 1, 0, 1, 0, 1, 0, 1],
            "unknown_loc": [2] * 8,
        }
    )
    result = compute_bbq_statistical_fairness(
        ["c"] * 4 + ["a", "b", "a", "b"],
        ["c"] * 8,
        metadata,
        min_group_count=4,
        require_all_contexts=True,
        max_accuracy_ci_width=1.0,
        min_disambig_coverage=0.70,
        min_disambig_coverage_lower=0.0,
    )
    assert result.ready
    assert np.isclose(result.loss, 1.0)
    assert not result.diagnostics["coverage_valid"]
    assert np.isclose(result.diagnostics["disambig_coverage"], 0.0)


def test_civil_equalized_odds_detects_identity_fpr_gap():
    rows = []
    y_true = []
    y_pred = []
    for identity in (1, 0):
        for _ in range(20):
            rows.append({"male": identity})
            y_true.append("toxic")
            y_pred.append("toxic")
        for index in range(20):
            rows.append({"male": identity})
            y_true.append("non_toxic")
            y_pred.append("toxic" if identity == 1 and index < 10 else "non_toxic")
    result = compute_civil_equalized_odds(
        y_true,
        y_pred,
        pd.DataFrame(rows),
        min_group_count=8,
        min_valid_identities=1,
        identity_columns=("male",),
        max_rate_ci_width=1.0,
    )
    assert result.ready
    assert result.loss > 0.20
    assert result.diagnostics["rates_by_identity"]["male"]["fpr_gap"] > 0.40


def test_bios_statistical_readiness_requires_support():
    metadata = pd.DataFrame({"gender": [0, 0, 1, 1]})
    result = compute_bios_statistical_tpr_gap(
        ["nurse"] * 4,
        ["nurse", "nurse", "nurse", "nurse"],
        metadata,
        min_group_count=4,
        min_valid_professions=1,
        max_rate_ci_width=1.0,
    )
    assert not result.ready
