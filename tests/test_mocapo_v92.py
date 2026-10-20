"""Behavioral contracts for the complete three-objective MO-CAPO extension."""

import importlib.util
import pickle
import random
from pathlib import Path

import numpy as np
import pytest
from promptolution.utils.prompt import Prompt

from tests.fakes import optimizer
from tri_fair.optimizers.mocapo_fair import MOCAPOFairOptimizer
from tri_fair.optimizers.pareto import ParetoPromptOptimizer
from tri_fair.runtime.budget import (
    BudgetAwarePredictor,
    BudgetExhausted,
    TokenBudgetController,
)
from tri_fair.runtime.checkpointing import (
    capture_optimizer_state,
    restore_optimizer_state,
)


def configured(**kwargs):
    random.seed(73)
    np.random.seed(73)
    x = optimizer(MOCAPOFairOptimizer, **kwargs)
    x.budget_controller = TokenBudgetController(
        llm=x.predictor.llm, max_tokens=100000, max_output_tokens=16
    )
    x.predictor = BudgetAwarePredictor(x.predictor, x.budget_controller)
    x._pre_optimization_loop()
    return x


def test_initialization_single_common_block():
    x = configured()
    assert len(x.initial_common_blocks) == 1
    assert all(x._blocks(p) == x.initial_common_blocks for p in x.prompts)
    assert len(x.prompts) == x.population_size


def test_original_capo_variation_without_adaptive_inheritance():
    assert (
        MOCAPOFairOptimizer._generate_challengers
        is ParetoPromptOptimizer._generate_challengers
    )
    assert not any(
        "Directional" in c.__name__ or "AdaptiveTri" in c.__name__
        for c in MOCAPOFairOptimizer.__mro__
    )


def test_survivors_need_common_basis_not_full_dataset():
    x = configured()
    child = Prompt("new challenger common coverage")
    x._do_intensification(child)
    assert x._blocks(child) == x.initial_common_blocks
    assert child in x.incumbents + x.non_incumbents
    assert len(x._blocks(child)) < x.task.n_blocks


def race(x, monkeypatch, ready=True, child=(0.5, -20, -0.4)):
    x.incumbents = x.incumbents[:1]
    x.non_incumbents = []
    monkeypatch.setattr(x, "_get_common_blocks", lambda ps: [0, 1])
    calls = []

    def vectors(prompts, blocks):
        calls.append(list(blocks))
        return np.array([[0.9, -10, -0.1]] * (len(prompts) - 1) + [child], float)

    monkeypatch.setattr(x, "_get_block_vectors", vectors)
    monkeypatch.setattr(x, "_supported", lambda ps, bs: ready)
    x.global_min_bounds = np.array([0, -100, -1.0])
    x.global_max_bounds = np.array([1, 0, 0.0])
    return calls


def test_rejected_child_stays_as_genetic_material(monkeypatch):
    x = configured()
    calls = race(x, monkeypatch)
    child = Prompt("dominated")
    x._do_intensification(child)
    assert len(calls) == 1 and child in x.non_incumbents and child not in x.incumbents
    assert x.algorithm_counts["early_rejections"] == 1


def test_unsupported_fairness_defers_rejection(monkeypatch):
    x = configured()
    calls = race(x, monkeypatch, ready=False)
    child = Prompt("unsupported")
    x._do_intensification(child)
    assert (
        len(calls) == 3
        and x.algorithm_counts["early_rejections"] == 0
        and child in x.incumbents
    )


def test_last_common_block_admits_then_reassesses_front(monkeypatch):
    x = configured()
    race(x, monkeypatch, child=(0.99, -20, -0.4))
    x._do_intensification(Prompt("tradeoff"))
    assert (
        x.algorithm_counts["common_basis_survivors"] == 1
        and x.algorithm_counts["early_rejections"] == 0
    )


def test_observed_normalization_uses_all_three_dimensions():
    x = configured()
    x.global_min_bounds = np.array([0.0, -100.0, -1.0])
    x.global_max_bounds = np.array([1.0, 0.0, 0.0])
    assert (
        x._get_closest_incumbent(
            np.array([0.8, -20.0, -0.2]),
            np.array([[0.1, -20.0, -0.9], [0.81, -30.0, -0.21]]),
        )
        == 1
    )


def test_pooled_metrics_and_cache_reuse():
    x = configured()
    p = x.prompts[0]
    v = x._get_block_vectors([p], [0, 1])
    spent = x.budget_controller.consumed_tokens
    np.testing.assert_allclose(v, x._get_block_vectors([p], [1, 0]))
    assert spent == x.budget_controller.consumed_tokens
    pooled = x.task.objectives_from_frame(
        x.task.cached_prediction_frame(p, [0, 1]),
        x.cost_per_input_token,
        x.cost_per_output_token,
    )
    np.testing.assert_allclose(v[0], pooled)


def test_population_retains_different_evaluation_levels():
    x = configured()
    x._step()
    x._step()
    assert len({len(x._blocks(p)) for p in x.prompts}) > 1
    assert (
        len(x.prompts) == x.population_size
        and x.algorithm_counts["incumbent_advances"] > 0
    )


def test_incumbent_gap_advancement_before_new_blocks():
    x = configured()
    a, b = x.prompts[:2]
    x._get_block_vectors([a, b], [0])
    x._get_block_vectors([a], [1])
    x.runhistory = {
        k: v
        for k, v in x.runhistory.items()
        if k[0] in {x._key(a), x._key(b)} and k[1] in [0, 1]
    }
    x.runhistory.pop((x._key(b), 1), None)
    x.incumbents = [a, b]
    x.non_incumbents = []
    x._advance_one_incumbent()
    assert x._blocks(b) == [0, 1] and x.algorithm_counts["gap_advances"] == 1


def test_new_block_advances_only_one_incumbent():
    x = configured()
    before = sum(len(x._blocks(p)) for p in x.incumbents)
    x._advance_one_incumbent()
    assert sum(len(x._blocks(p)) for p in x.incumbents) == before + 1
    assert x.algorithm_counts["new_block_advances"] == 1


def selection(x, monkeypatch, blocks, ready=True):
    a, b = x.prompts[:2]
    x.incumbents = []
    x.non_incumbents = [a, b]
    monkeypatch.setattr(x, "_blocks", lambda p: blocks[0] if p == a else blocks[1])
    calls = []
    monkeypatch.setattr(
        x,
        "_get_block_vectors",
        lambda ps, bs: (
            calls.append(bs) or np.array([[0.7, -20, -0.3], [0.8, -19, -0.2]])
        ),
    )
    monkeypatch.setattr(x, "_supported", lambda ps, bs: ready)
    monkeypatch.setattr(random, "sample", lambda pool, n: [a, b])
    return a, b, calls


def test_weaker_dominance_uses_less_evaluated_subset(monkeypatch):
    x = configured()
    a, b, calls = selection(x, monkeypatch, ([0], [0, 1]))
    assert x._select_parent_from_pool([a, b]) == b and calls == [[0]]


def test_equal_block_sets_ignore_insertion_order(monkeypatch):
    x = configured()
    a, b, _ = selection(x, monkeypatch, ([0, 1], [1, 0]))
    assert x._select_parent_from_pool([a, b]) == b


def test_unsupported_weaker_dominance_is_deferred(monkeypatch):
    x = configured()
    a, b, _ = selection(x, monkeypatch, ([0], [0, 1]), False)
    x._select_parent_from_pool([a, b])
    assert x.algorithm_counts["weaker_dominance_checks"] == 0


def test_heterogeneous_survival_removes_fewest_evaluations(monkeypatch):
    x = configured()
    a, b, c = x.prompts[:3]
    x.population_size = 2
    x.incumbents = [a]
    x.non_incumbents = [b, c]
    monkeypatch.setattr(x, "_blocks", lambda p: [0] if p == b else [0, 1])
    x._select_survivors()
    assert x.non_incumbents == [c]


def test_budget_interrupt_keeps_valid_incumbents():
    x = configured()
    before = list(x.incumbents)
    x.budget_controller.max_tokens = x.budget_controller.consumed_tokens + 1
    with pytest.raises(BudgetExhausted):
        x._do_intensification(Prompt("unaffordable"))
    assert (
        x.incumbents == before
        and x.budget_controller.consumed_tokens <= x.budget_controller.max_tokens
    )


def test_checkpoint_resume_matches_uninterrupted_search():
    x = configured()
    x._step()
    state = pickle.loads(pickle.dumps(capture_optimizer_state(x)))
    x._step()
    expected = [p.construct_prompt() for p in x.prompts]
    y = configured()
    restore_optimizer_state(y, state)
    y._step()
    assert [p.construct_prompt() for p in y.prompts] == expected
    assert (
        y.runhistory.keys() == x.runhistory.keys()
        and y.algorithm_counts == x.algorithm_counts
    )


def test_checkpoint_rejects_changed_algorithm():
    x = configured()
    y = configured(no_weaker_dominance=True)
    with pytest.raises(ValueError, match="configuration"):
        y.load_optimizer_state_dict(x.optimizer_state_dict())


def test_distinct_parent_tournament_is_bounded(monkeypatch):
    x = configured()
    monkeypatch.setattr(x, "_select_parent_from_pool", lambda pool: pool[0])
    a, b = x._tournament_selection()
    assert a != b


@pytest.mark.parametrize(
    "method",
    ["Tri-Fair-v9.2", "Tri-Fair-v9.2-NoWeakerDominance", "Tri-Fair-v9.2-FullEval"],
)
def test_public_runner_and_frozen_audit(method, tmp_path, monkeypatch):
    from tests.test_integration import test_complete_public_runner_with_fake_models

    test_complete_public_runner_with_fake_models(
        method, tmp_path, monkeypatch, audit=True
    )


@pytest.mark.parametrize("profile,runs", [("screen", 20), ("ablation", 12)])
def test_launch_profiles(monkeypatch, profile, runs):
    monkeypatch.setenv("V92_PROFILE", profile)
    monkeypatch.setenv("PILOT_SEEDS", "72")
    p = Path(__file__).resolve().parents[1] / "research/run_pilot_v92.py"
    spec = importlib.util.spec_from_file_location("plan92", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    plan = mod.plan()
    assert (
        plan["optimizer_runs"] == runs
        and not plan["pilot_settings"]["reclaim_audit_reserve"]
    )
    assert plan["mixed_fidelity_population"] and plan["max_concurrent_gpus"] == 4


def test_full_evaluation_is_separate_explicit_ablation():
    x = configured(full_evaluation=True)
    assert all(len(x._blocks(p)) == x.task.n_blocks for p in x.prompts)
    assert not x.search_diagnostics()["progressive_intensification"]


def test_final_only_renderer_exports_no_fabricated_trajectory(tmp_path):
    import pandas as pd

    from tests.test_research_figures import rows

    path = Path(__file__).resolve().parents[1] / "research/plot_v92.py"
    spec = importlib.util.spec_from_file_location("plots92", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    frame = rows().replace({"optimizer": {"Tri-Fair-v8.6": "Tri-Fair-v9.2"}})
    frame["validation_input_tokens"] = 50.0
    frame["validation_output_tokens"] = 2.0
    dest = tmp_path / "heldout_validation/qwen-3-30b/winobias/seed72"
    dest.mkdir(parents=True)
    frame.to_parquet(dest / "eval_checkpoints_union.parquet", index=False)
    out = tmp_path / "figures"
    made = mod.render(tmp_path, out, budget=1000)
    assert made == [
        "final_front_indicators",
        "balanced_objectives",
        "fewshot_cost_quality",
    ]
    assert (out / "final_front_indicators.png").is_file()
    exported = pd.read_csv(out / "measured_final_indicators.csv")
    assert len(exported) == 2 and not list(out.glob("*trajectory*"))


def test_reclaim_flag_rejected_before_model_loading():
    from types import SimpleNamespace

    from tri_fair.cli.train import _validate_args

    with pytest.raises(ValueError, match="frozen terminal audit"):
        _validate_args(
            SimpleNamespace(optimizer="Tri-Fair-v9.2", reclaim_audit_reserve=True)
        )


def test_main_variation_cannot_silently_enable_adaptive_edits():
    with pytest.raises(ValueError, match="original CAPO variation"):
        configured(objective_aware_variation=True)


def test_frozen_audit_does_not_modify_search_population(tmp_path):
    from tri_fair.runtime.finalization import finalize_search_front

    x = configured()
    x._step()
    keys = [p.construct_prompt() for p in x.prompts]
    levels = {p.construct_prompt(): x._blocks(p) for p in x.prompts}
    result = finalize_search_front(x, tmp_path, count=6, total_budget=100000)
    assert result["status"] == "complete" and not result["feedback_to_optimizer"]
    assert keys == [p.construct_prompt() for p in x.prompts]
    assert levels == {p.construct_prompt(): x._blocks(p) for p in x.prompts}


def test_full_evaluation_ablation_never_performs_partial_racing():
    x = configured(full_evaluation=True)
    child = Prompt("full-evaluation child")
    x._do_intensification(child)
    assert len(x._blocks(child)) == x.task.n_blocks
    assert x.algorithm_counts["nearest_comparisons"] == 0
    assert x.algorithm_counts["early_rejections"] == 0
    assert x.algorithm_counts["full_evaluation_challengers"] == 1
