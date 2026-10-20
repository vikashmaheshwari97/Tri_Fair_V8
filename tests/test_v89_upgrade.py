"""Offline tests for real admission regressions, strategy credit and frozen launch."""

import importlib.util
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
from promptolution.utils.prompt import Prompt

from tests.fakes import optimizer
from tri_fair.optimizers.adaptive import (
    AdaptiveTriFairOptimizer,
    contribution_reward,
    control_evidence,
)
from tri_fair.optimizers.budget_fill import FILLED_BASELINES, BudgetFilledNSGAII
from tri_fair.optimizers.baselines import SPEA2POFair, NSGAIIIPOFair, RandomSearchPOFair
from tri_fair.optimizers.nsga2 import NSGA2Fair
from tri_fair.runtime.budget import (
    TokenBudgetController,
    BudgetAwarePredictor,
    BudgetExhausted,
)
from tri_fair.runtime.checkpointing import (
    capture_optimizer_state,
    restore_optimizer_state,
)
from tri_fair.runtime.finalization import reclaim_and_finalize

ROOT = Path(__file__).resolve().parents[1]


def configured(cls=BudgetFilledNSGAII, cap=100000, **kwargs):
    if cls is AdaptiveTriFairOptimizer:
        kwargs.setdefault("repeat_check_examples", 0)
    obj = optimizer(cls, **kwargs)
    obj.crossovers_per_iter = 1
    c = TokenBudgetController(
        llm=obj.predictor.llm, max_tokens=cap, max_output_tokens=16
    )
    obj.budget_controller = c
    obj.predictor = BudgetAwarePredictor(obj.predictor, c)
    obj._pre_optimization_loop()
    return obj, c


def test_reproduces_empty_as_zero_estimate_budget_bug_in_old_path():
    obj, c = configured(NSGA2Fair)
    obj._generate_challengers = lambda: []
    with pytest.raises(BudgetExhausted) as caught:
        obj._step()
    assert caught.value.estimated_tokens == 0
    assert c.remaining_tokens > 50000


def test_empty_batches_are_retried_without_fake_expenditure(tmp_path):
    obj, c = configured()
    obj.set_artifact_directory(tmp_path)
    child = Prompt("A novel complete decision rule")
    batches = iter([[], [], [child]])
    obj._generate_challengers = lambda: next(batches)
    before = c.consumed_tokens
    obj._step()
    assert c.consumed_tokens > before and not c.stop_requested
    assert obj.fill_counts["empty_proposal_batches"] == 2
    assert obj.fill_counts["evaluated_candidates"] == 1
    assert set(obj.task.get_evaluated_blocks([child])[child]) == set(
        range(obj.task.n_blocks)
    )
    assert [
        json.loads(s)["status"]
        for s in (tmp_path / "proposal_admission.jsonl").read_text().splitlines()
    ] == ["empty", "empty", "admitted"]


def test_cached_candidate_is_not_counted_as_new_budget_use():
    obj, c = configured()
    cached = obj.prompts[0]
    child = Prompt("An unseen different decision rule")
    batches = iter([[cached], [child]])
    obj._generate_challengers = lambda: next(batches)
    assert obj._filled_challengers() == [child]
    assert obj.fill_counts["cached_or_duplicate_candidates"] == 1


def test_unaffordable_prefix_does_not_hide_later_affordable_candidate():
    obj, c = configured()
    expensive = Prompt("long " * 1000)
    cheap = Prompt("classify")
    c.max_tokens = c.consumed_tokens + obj._estimate_complete(cheap) + 10
    obj._generate_challengers = lambda: [expensive, cheap]
    assert obj._filled_challengers() == [cheap]
    assert obj.fill_counts["unaffordable_candidates"] == 1
    assert not c.stop_requested


def test_proposal_failure_is_not_claimed_as_budget_exhaustion():
    obj, c = configured(proposal_retry_limit=2)
    obj._generate_challengers = lambda: []
    before = c.consumed_tokens
    with pytest.raises(
        BudgetExhausted, match="proposal_retry_limit_no_novel_candidate"
    ):
        obj._step()
    assert c.consumed_tokens == before and not c.stop_requested
    assert c.stopping_reason is None
    assert obj.runtime_stop_reason == "proposal_retry_limit_no_novel_candidate"


def test_real_affordability_stop_has_nonzero_observed_estimate():
    obj, c = configured(proposal_retry_limit=2)
    child = Prompt("long " * 2000)
    obj._generate_challengers = lambda: [child]
    with pytest.raises(BudgetExhausted):
        obj._step()
    assert c.stopping_reason == "next_complete_candidate_exceeds_budget"
    assert c.estimated_next_operation_tokens > c.remaining_tokens
    assert obj.fill_trace[-1]["global_infeasibility_proved"] is False


@pytest.mark.parametrize("cls", [AdaptiveTriFairOptimizer, *FILLED_BASELINES.values()])
def test_every_method_gets_full_coverage_and_hard_cap(cls, tmp_path):
    random.seed(72)
    np.random.seed(72)
    obj, c = configured(cls, cap=25000)
    obj.set_artifact_directory(tmp_path)
    obj._resume_loaded = True
    obj.optimize(100)
    final = reclaim_and_finalize(
        obj, tmp_path, count=4, total_budget=45000, max_steps=100
    )
    assert c.consumed_tokens <= 45000
    assert c.consumed_tokens >= 30000  # meaningful work, not padding to exactly 100%
    assert final["status"] == "complete"
    assert final["audit_downstream_tokens"] == 0
    for p in obj.prompts:
        assert set(obj.task.get_evaluated_blocks([p])[p]) == set(
            range(obj.task.n_blocks)
        )
    assert obj.fill_counts["evaluated_candidates"] > 0


@pytest.mark.parametrize(
    "name,base",
    [
        ("NSGAII-PO-Fair", NSGA2Fair),
        ("NSGAIII-PO-Fair", NSGAIIIPOFair),
        ("SPEA2-PO-Fair", SPEA2POFair),
        ("RandomSearch-PO-Fair", RandomSearchPOFair),
    ],
)
def test_named_baselines_keep_selectors_and_proposal_implementations(name, base):
    cls = FILLED_BASELINES[name]
    assert cls._generate_challengers is base._generate_challengers
    assert cls._environmental_selection is base._environmental_selection
    assert cls._tournament_select_one is base._tournament_select_one


def test_original_protected_baseline_data_and_metrics_hashes_unchanged():
    from tri_fair.cli.preflight_v88 import check

    assert check()["ok"]


def test_reward_uses_measured_front_gain_and_tokens_not_intended_mode():
    old = [[0.3, 0.3, 0.3]]
    assert contribution_reward(old, [0.4, 0.4, 0.4], 10000) == (0, 0)
    reward, gain = contribution_reward(old, [0.2, 0.2, 0.2], 200000)
    assert reward == 0.5 and gain > 0
    assert contribution_reward(old, [0.2, 2, 0.2], 1000) == (0, 0)


def test_worsening_fairness_edit_receives_zero_fairness_credit(monkeypatch):
    obj, _ = configured(AdaptiveTriFairOptimizer)
    parent = obj.prompts[0]
    child = Prompt("Novel repaired classifier")
    obj.task.evaluate([child], obj.predictor, eval_strategy="full")
    old = dict(quality=0.8, cost=20.0, unfairness=0.2)
    new = dict(quality=0.9, cost=15.0, unfairness=0.3)
    monkeypatch.setattr(obj, "_guard_record", lambda p: new if p == child else old)
    monkeypatch.setattr(obj, "_update_archive", lambda *args: None)
    monkeypatch.setattr(obj, "_fold_consistency", lambda *args: (1.0, 4))
    obj._pending_v89 = {
        child.construct_prompt(): (
            child,
            parent,
            "fairness",
            "intended improvement",
            np.array([[0.2, 0.3, 0.2]]),
        )
    }
    obj._last_candidate_token_deltas = {child.construct_prompt(): 1000}
    obj._on_filled_evaluation([child], 1000)
    assert obj.v89_outcomes[-1]["unfairness_change"] > 0
    assert obj.v89_outcomes[-1]["strategy_reward"] == 0
    assert obj.strategy_state["fairness"]["beta"] == 2


def test_controls_and_donor_read_cached_search_only():
    obj, c = configured(AdaptiveTriFairOptimizer)
    before = c.consumed_tokens
    parent = obj.prompts[0]
    controls = control_evidence(obj.task, parent)
    assert controls and any(r["correct"] for r in controls)
    obj._donor(parent)
    assert c.consumed_tokens == before


def test_unsupported_fold_consistency_is_missing_not_zero(monkeypatch):
    obj, _ = configured(AdaptiveTriFairOptimizer)
    monkeypatch.setattr(
        obj.task, "objectives_from_frame", lambda *args: np.full(3, np.nan)
    )
    assert obj._fold_consistency(obj.prompts[0], obj.prompts[1]) == (None, 0)


def test_checkpoint_restores_posterior_seed_stream_archive_and_next_proposal():
    random.seed(72)
    np.random.seed(72)
    obj, _ = configured(AdaptiveTriFairOptimizer)
    obj._resume_loaded = True
    obj.optimize(3)
    state = capture_optimizer_state(obj)
    expected = obj._filled_challengers()
    other, _ = configured(AdaptiveTriFairOptimizer)
    restore_optimizer_state(other, state)
    actual = other._filled_challengers()
    assert [p.construct_prompt() for p in expected] == [
        p.construct_prompt() for p in actual
    ]
    assert other.strategy_state == state["optimizer_extra"]["fields"]["strategy_state"]
    assert other.fill_attempts == obj.fill_attempts


def test_pilot_uses_same_explicit_fill_file_for_every_method(capsys):
    from tri_fair.cli.pilot_v89 import main

    assert main(["--dataset", "bbq"]) == 0
    lines = [
        s for s in capsys.readouterr().out.splitlines() if "tri_fair.cli.train" in s
    ]
    assert len(lines) == 5
    for line in lines:
        assert "--budget-per-run 2000000" in line
        assert "--optimizer-overrides" in line and "v89_budget_fill.json" in line
    assert json.loads((ROOT / "configs/v89_budget_fill.json").read_text()) == {
        "budget_fill": True
    }


@pytest.mark.parametrize("method", ["Tri-Fair-v8.9", *FILLED_BASELINES])
def test_registry_really_constructs_filled_classes(method):
    from tests.fakes import task_and_pool, FakeLLM, FakePredictor
    from tri_fair.config.registry import build_optimizer
    from tri_fair.config.study import build_optimizer_registry
    from tri_fair.config.models import ALL_MODELS

    task, pool = task_and_pool(eval_strategy="full")
    args = SimpleNamespace(
        optimizer_overrides={"budget_fill": True},
        crossovers_per_iteration=1,
        max_few_shot_examples=2,
        disable_objective_aware_variation=False,
        input_cost_multiplier=1.0,
        output_cost_multiplier=1.0,
    )
    result = build_optimizer(
        args=args,
        optimizer_config=build_optimizer_registry()[method],
        model_config=ALL_MODELS["qwen-3-30b"],
        predictor=FakePredictor(),
        task=task,
        meta_llm=FakeLLM(),
        initial_prompts=["rule A", "rule B"],
        df_fewshots=pool,
        callbacks=[],
    )
    cls = (
        AdaptiveTriFairOptimizer
        if method == "Tri-Fair-v8.9"
        else FILLED_BASELINES[method]
    )
    assert type(result) is cls
    assert result.effective_parameters["budget_fill"] is True


@pytest.mark.parametrize("method", ["Tri-Fair-v8.9", *FILLED_BASELINES])
def test_public_train_runner_with_filled_protocol(method, tmp_path, monkeypatch):
    import tri_fair.cli.train as runner
    from tests.test_integration import test_complete_public_runner_with_fake_models

    original = runner.parse_args

    def parsed():
        args = original()
        args.optimizer_overrides = {"budget_fill": True}
        return args

    monkeypatch.setattr(runner, "parse_args", parsed)
    test_complete_public_runner_with_fake_models(method, tmp_path, monkeypatch)


def load_research(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "research" / (name + ".py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_single_seed_then_three_seed_plan(monkeypatch):
    monkeypatch.setenv("PILOT_SEEDS", "72")
    module = load_research("run_pilot_v89")
    plan = module.plan()
    assert plan["task_count"] == 4 and plan["optimizer_runs"] == 20
    assert plan["max_concurrent_gpus"] == 4
    assert module.task(2) == ("bias_in_bios", 72)
    monkeypatch.setenv("PILOT_SEEDS", "72,73,74")
    module = load_research("run_pilot_v89")
    assert module.plan()["optimizer_runs"] == 60
    assert module.plan()["max_concurrent_gpus"] == 5
    monkeypatch.setenv("PILOT_BUDGET", "5000000")
    with pytest.raises(ValueError, match="2000000"):
        module.plan()


def test_observed_plot_support_is_not_extrapolated_or_made_monotone():
    import pandas as pd

    module = load_research("plot_v89")
    frame = pd.DataFrame(
        dict(
            seed=[72, 72, 73, 73],
            actual_budget_tokens=[100000, 500000, 200000, 400000],
            value=[1.0, 3.0, 2.0, 0.0],
        )
    )
    x, mean, sd = module.observed_summary(frame, "value")
    assert x.tolist() == [0.2, 0.4] and mean.tolist() == [1.5, 0.5]
    assert sd is not None
    _, _, single_sd = module.observed_summary(frame[frame.seed == 72], "value")
    assert single_sd is None


def test_single_seed_table_does_not_invent_zero_standard_deviation(monkeypatch):
    import pandas as pd

    monkeypatch.setenv("PILOT_SEEDS", "72")
    module = load_research("run_pilot_v89")
    frame = pd.DataFrame(
        dict(
            dataset=["bbq"],
            model=["qwen-3-30b"],
            optimizer=["Tri-Fair-v8.9"],
            budget_checkpoint=[2000000],
            seed=[72],
            actual_budget_tokens=[1900000],
            hv=[0.8],
        )
    )
    summary = module.descriptive_summary(frame)
    assert summary.hv_mean.iloc[0] == 0.8 and pd.isna(summary.hv_std.iloc[0])
