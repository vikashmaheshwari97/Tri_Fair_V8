"""Behavioral regressions for disparity targeting, coverage and measured credit."""

import importlib.util
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from promptolution.utils.prompt import Prompt

from tests.fakes import optimizer
from tri_fair.optimizers.directional import STRATEGIES, DirectionalTriFairOptimizer
from tri_fair.runtime.budget import BudgetAwarePredictor, TokenBudgetController
from tri_fair.runtime.checkpointing import (
    capture_optimizer_state,
    restore_optimizer_state,
)
from tri_fair.search.disparity import (
    confusion_demonstrations,
    coverage_select,
    disparity_evidence,
)

ROOT = Path(__file__).resolve().parents[1]


def configured(**kwargs):
    obj = optimizer(DirectionalTriFairOptimizer, repeat_check_examples=0, **kwargs)
    obj.budget_controller = TokenBudgetController(
        llm=obj.predictor.llm, max_tokens=100000, max_output_tokens=16
    )
    obj.predictor = BudgetAwarePredictor(obj.predictor, obj.budget_controller)
    obj._pre_optimization_loop()
    return obj


def test_bios_targets_gap_not_uniformly_high_error_group():
    frame = pd.DataFrame(
        [
            dict(
                _review_id=str(i),
                input="person",
                _review_target=label,
                _review_prediction=pred,
                gender=g,
            )
            for i, (label, pred, g) in enumerate(
                [
                    ("nurse", "teacher", 0),
                    ("nurse", "teacher", 1),
                    ("teacher", "nurse", 1),
                    ("teacher", "teacher", 1),
                    ("teacher", "teacher", 0),
                    ("teacher", "teacher", 0),
                ]
            )
        ]
    )
    task = SimpleNamespace(
        n_blocks=2,
        fairness_metric_name="bios_tpr_gap_statistical",
        x_column="input",
        protected_columns=["gender"],
        cached_prediction_frame=lambda *a: frame,
        get_last_record=lambda *a: {
            "diagnostics": {"tpr_gap_by_profession": {"nurse": 0.0, "teacher": 0.5}}
        },
    )
    result = disparity_evidence(task, None)
    assert result["selected_component"]["label"] == "teacher"
    assert result["selected_component"]["repair_group"] == 1
    assert all(r["gold"] == "teacher" for r in result["cases"])
    assert {r["correct"] for r in result["cases"]} == {True, False}


def test_metric_evidence_targets_binding_utility_penalty():
    frame = pd.DataFrame(
        [
            dict(
                _review_id="0",
                input="comment",
                _review_target="toxic",
                _review_prediction="non_toxic",
                male=1,
            )
        ]
    )
    task = SimpleNamespace(
        n_blocks=1,
        fairness_metric_name="civil_equalized_odds",
        x_column="input",
        protected_columns=["male"],
        cached_prediction_frame=lambda *a: frame,
        get_last_record=lambda *a: {
            "diagnostics": {
                "utility_penalty": 0.5,
                "equalized_odds_rms": 0.1,
                "class_recall": {"toxic": 0.25, "non_toxic": 0.95},
            }
        },
    )
    result = disparity_evidence(task, None)
    assert result["selected_component"]["metric"] == "binding_utility_safeguard"
    assert result["selected_component"]["label"] == "toxic"


@pytest.mark.parametrize(
    "kind,gap,expected",
    [("tpr", 0.3, 0), ("tpr", -0.3, 1), ("fpr", 0.3, 1), ("fpr", -0.3, 0)],
)
def test_equalized_odds_repairs_correct_side(kind, gap, expected):
    target = "toxic" if kind == "tpr" else "non_toxic"
    wrong = "non_toxic" if kind == "tpr" else "toxic"
    frame = pd.DataFrame(
        [
            dict(
                _review_id=str(g),
                input="comment",
                _review_target=target,
                _review_prediction=wrong if g == expected else target,
                male=g,
            )
            for g in [0, 1]
        ]
    )
    task = SimpleNamespace(
        n_blocks=2,
        fairness_metric_name="civil_equalized_odds",
        x_column="input",
        protected_columns=["male"],
        cached_prediction_frame=lambda *a: frame,
        get_last_record=lambda *a: {
            "diagnostics": {
                "rates_by_identity": {
                    "male": {
                        "tpr_gap": gap if kind == "tpr" else 0.0,
                        "fpr_gap": gap if kind == "fpr" else 0.0,
                    }
                }
            }
        },
    )
    result = disparity_evidence(task, None)
    assert result["selected_component"]["repair_group"] == expected


def test_coverage_retains_balanced_and_all_three_anchors():
    losses = np.array(
        [
            [0.1, 0.8, 0.8],
            [0.8, 0.1, 0.8],
            [0.8, 0.8, 0.1],
            [0.35, 0.35, 0.35],
            [0.36, 0.36, 0.36],
        ]
    )
    assert set(coverage_select(losses, 4)) == {0, 1, 2, 3}


def test_coverage_does_not_admit_dominated_or_nonfinite_point():
    assert coverage_select([[0.2, 0.2, 0.2], [0.3, 0.3, 0.3]], 1) == [0]
    with pytest.raises(ValueError):
        coverage_select([[float("nan"), 0, 0]], 1)


def test_forced_operator_exploration_survives_bad_early_rewards():
    obj = configured()
    for state in obj.strategy_state.values():
        state.update(evaluations=10, reward_sum=0)
    obj.strategy_state["compress"]["reward_sum"] = 10
    seen = []
    for i in range(12):
        obj.proposal_round = i
        seen.append(obj._choose_strategy())
    assert set(seen[::2]) == set(STRATEGIES)


def test_demonstrations_can_reach_beyond_old_first_four_examples():
    obj = configured()
    extra = obj.df_few_shots.copy()
    extra["input"] = extra["input"] + " another example"
    obj.df_few_shots = pd.concat([obj.df_few_shots, extra], ignore_index=True)
    variants = {
        tuple(
            confusion_demonstrations(
                obj, {"confusions": [dict(gold="nurse", predicted="teacher")]}, offset=i
            )
        )
        for i in range(16)
    }
    assert len(variants) > 4
    for demos in variants:
        assert len(demos) == 2 and any("<final_answer>nurse" in d for d in demos)


def test_full_eval_budget_checkpoint_and_next_proposal(tmp_path):
    random.seed(72)
    np.random.seed(72)
    obj = configured()
    obj.set_artifact_directory(tmp_path)
    obj._resume_loaded = True
    obj.optimize(3)
    assert obj.fill_counts["evaluated_candidates"] > 0
    assert obj.budget_controller.consumed_tokens <= obj.budget_controller.max_tokens
    assert set(obj.strategy_state) == set(STRATEGIES)
    for p in obj.prompts:
        assert set(obj.task.get_evaluated_blocks([p])[p]) == set(
            range(obj.task.n_blocks)
        )
    state = capture_optimizer_state(obj)
    expected = obj._filled_challengers()
    restored = configured()
    restore_optimizer_state(restored, state)
    actual = restored._filled_challengers()
    assert [p.construct_prompt() for p in actual] == [
        p.construct_prompt() for p in expected
    ]
    assert restored.strategy_state == obj.strategy_state
    assert restored.coverage_turn == obj.coverage_turn


def test_continuous_reward_and_batch_credit(monkeypatch):
    obj = configured()
    parent = obj.prompts[0]
    children = [Prompt("child A"), Prompt("child B")]
    old = dict(quality=0.7, cost=30.0, unfairness=0.3)
    new = dict(quality=0.7001, cost=30.0, unfairness=0.3)
    monkeypatch.setattr(obj, "_guard_record", lambda p: old if p == parent else new)
    monkeypatch.setattr(obj, "_fold_consistency", lambda *a: (None, 0))
    monkeypatch.setattr(obj, "_update_archive", lambda *a: None)
    obj._pending_v89 = {
        p.construct_prompt(): (p, parent, "repair", "test", np.array([obj._loss(old)]))
        for p in children
    }
    obj._last_candidate_token_deltas = {p.construct_prompt(): 100 for p in children}
    obj._on_filled_evaluation(children, 200)
    assert 0 < obj.v89_outcomes[-2]["strategy_reward"] < 0.1
    assert obj.v89_outcomes[-1]["strategy_reward"] == 0


def test_no_extra_inference_for_disparity_evidence():
    obj = configured()
    before = obj.budget_controller.consumed_tokens
    disparity_evidence(obj.task, obj.prompts[0])
    assert obj.budget_controller.consumed_tokens == before


def test_meta_request_prunes_context_but_keeps_metric_comparison():
    obj = configured(generation_input_token_limit=900)
    evidence = {
        "evidence": [{"input": "long " * 300} for _ in range(4)],
        "controls_and_errors": [{"input": "other " * 300} for _ in range(4)],
        "metric_aligned": {
            "cases": [{"input": "case " * 20}, {"input": "control " * 20}]
        },
    }
    request = obj._request(obj.prompts[0], "fairness", evidence)
    assert len(obj.meta_llm.tokenizer.encode(request)) <= 900
    assert len(evidence["metric_aligned"]["cases"]) == 2


def test_checkpoint_rejects_changed_ablation():
    obj = configured()
    other = configured(enable_disparity_evidence=False)
    with pytest.raises(ValueError, match="Directional Tri-Fair"):
        other.load_optimizer_state_dict(obj.optimizer_state_dict())


@pytest.mark.parametrize(
    "method", ["Tri-Fair-v9.0", "Tri-Fair-v9.0-NoDisparity", "Tri-Fair-v9.0-NoCoverage"]
)
def test_public_train_runner(method, tmp_path, monkeypatch):
    from tests.test_integration import test_complete_public_runner_with_fake_models

    test_complete_public_runner_with_fake_models(method, tmp_path, monkeypatch)


def load_launcher():
    spec = importlib.util.spec_from_file_location(
        "run_pilot_v90", ROOT / "research/run_pilot_v90.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("profile,runs", [("ablation", 16), ("screen", 24)])
def test_launch_plan_has_correct_methods_budget_and_model(monkeypatch, profile, runs):
    monkeypatch.setenv("V90_PROFILE", profile)
    monkeypatch.setenv("PILOT_SEEDS", "72")
    monkeypatch.setenv("MODEL", "mistral-3-24b")
    plan = load_launcher().plan()
    assert plan["optimizer_runs"] == runs
    assert plan["downstream_ceiling"] == runs * 2000000
    assert plan["model"] == "mistral-3-24b"
    assert not plan["final_test_used"]


def test_pilot_shared_budget_and_historical_control(capsys):
    from tri_fair.cli.pilot_v90 import main

    assert main(["--dataset", "bias_in_bios"]) == 0
    lines = [
        s for s in capsys.readouterr().out.splitlines() if "tri_fair.cli.train" in s
    ]
    assert len(lines) == 6 and any("Tri-Fair-v8.9" in line for line in lines)
    assert all(
        "--budget-per-run 2000000" in line and "v89_budget_fill.json" in line
        for line in lines
    )


def test_protected_comparators_and_metrics_unchanged():
    from tri_fair.cli.preflight_v88 import check

    assert check()["ok"]
