"""Plots must preserve measured identities and omit unavailable trajectories."""

import pandas as pd
from tri_fair.reporting.research_figures import (
    plot_research,
    trajectory_rows,
    final_rows,
)
from tri_fair.reporting.front_report import analyse
from tri_fair.reporting.research_figures import attainment_curve


def rows():
    return pd.DataFrame(
        [
            dict(
                dataset="winobias",
                model="qwen-3-30b",
                optimizer=m,
                seed=72,
                prompt_id=f"{m}-{i}",
                prompt=f"{m}-{i}",
                few_shots_json="[]",
                chosen_step=3,
                actual_budget_tokens=900,
                evaluation_split="validation",
                selection_policy="audited_final_shortlist",
                dev_quality=0.8 + i * 0.05,
                dev_cost=10 + i * 5,
                dev_fairness=0.2 - i * 0.03,
                dev_fairness_ready=True,
                validation_quality=0.78 + i * 0.05,
                validation_cost=10 + i * 5,
                validation_fairness=0.21 - i * 0.03,
                validation_fairness_ready=True,
            )
            for m in ["Tri-Fair-v8.6", "NSGAII-PO-Fair"]
            for i in range(2)
        ]
    )


def test_final_only_data_cannot_generate_a_fake_optimization_trajectory(tmp_path):
    frame = rows()
    before = frame.copy(deep=True)
    protocol = plot_research(
        frame, tmp_path, split="validation", budget=1000, formats=("svg",)
    )
    assert "optimization_trajectories" not in protocol["figures"]
    assert protocol["seeds"] == [72]
    assert (tmp_path / "objective_projections.svg").is_file()
    assert (tmp_path / "prompt_composition.svg").is_file()
    pd.testing.assert_frame_equal(frame, before)


def test_audit_and_history_stay_distinct_for_final_indicators():
    frame = rows()
    history = frame.copy()
    history["selection_policy"] = "current_incumbents"
    history["dev_quality"] = 0.65
    combined = pd.concat([history, frame], ignore_index=True)
    result = analyse(combined, split="validation", budget=1000)
    assert result.n_candidates.eq(2).all()
    assert len(final_rows(combined)) == 4
    trajectories = trajectory_rows(combined, split="validation", budget=1000)
    assert len(trajectories) == 2 and set(trajectories.actual_budget_tokens) == {900}


def test_attainment_has_no_solution_below_cost_or_fairness_limits():
    import numpy as np

    frame = rows().iloc[:2]
    values = attainment_curve(frame, [5, 10, 15], split="validation", threshold=0.30)
    assert np.isnan(values[0])
    np.testing.assert_allclose(values[1:], [0.78, 0.83])
    assert np.isnan(
        attainment_curve(frame, [10, 15], split="validation", threshold=0.10)
    ).all()
