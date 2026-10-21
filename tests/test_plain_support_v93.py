"""Plain baseline identity and unchanged-threshold readiness warm-up contracts."""

import importlib.util
import pickle
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.fakes import FakeLLM, FakePredictor, optimizer, task_and_pool
from tri_fair.optimizers.baselines import NSGAIIIPOFair, RandomSearchPOFair, SPEA2POFair
from tri_fair.optimizers.mocapo_fair import MOCAPOFairOptimizer
from tri_fair.optimizers.nsga2 import NSGA2Fair
from tri_fair.optimizers.plain_baselines import (
    PLAIN_BASELINES,
    PlainNSGAII,
    PlainNSGAIII,
    PlainRandomSearch,
    PlainSPEA2,
)
from tri_fair.optimizers.support_ready import SupportReadyTriFairOptimizer
from tri_fair.runtime.budget import (
    BudgetAwarePredictor,
    BudgetExhausted,
    TokenBudgetController,
)
from tri_fair.runtime.checkpointing import (
    capture_optimizer_state,
    restore_optimizer_state,
)


def ready_optimizer(cls=SupportReadyTriFairOptimizer, **kwargs):
    random.seed(72)
    np.random.seed(72)
    task, pool = task_and_pool()
    task.min_group_count = 6  # two examples per gender/profession/block: need3 blocks
    obj = cls(
        task=task,
        predictor=FakePredictor(),
        meta_llm=FakeLLM(),
        initial_prompts=[f"Use professional evidence. Seed{i}" for i in range(4)],
        df_few_shots=pool,
        upper_shots=1,
        crossovers_per_iter=1,
        create_fs_reasoning=False,
        cost_per_input_token=0.11,
        cost_per_output_token=0.41,
        **kwargs,
    )
    obj.budget_controller = TokenBudgetController(
        llm=obj.predictor.llm, max_tokens=100000, max_output_tokens=16
    )
    obj.predictor = BudgetAwarePredictor(obj.predictor, obj.budget_controller)
    return obj


@pytest.mark.parametrize("cls", PLAIN_BASELINES.values())
def test_plain_baselines_never_call_racing_or_advancement(cls, monkeypatch):
    obj = optimizer(cls)

    def forbidden(*a, **k):
        raise AssertionError("Full-evaluation baseline called mixed-fidelity mechanism")

    for name in [
        "_do_intensification",
        "_advance_one_incumbent",
        "_select_parent_from_pool",
        "_get_common_blocks",
    ]:
        monkeypatch.setattr(obj, name, forbidden)
    monkeypatch.setattr(
        "tri_fair.optimizers.nsga2.generate_objective_aware_challengers", forbidden
    )
    obj.optimize(2)
    assert not obj.objective_aware_variation
    assert all(
        len(obj.task.get_evaluated_blocks([p])[p]) == obj.task.n_blocks
        for p in obj.prompts
    )
    for name in [
        "progressive_intensification",
        "mixed_fidelity_population",
        "weaker_dominance_selection",
        "incumbent_advancement",
    ]:
        assert obj.search_diagnostics()[name] is False


@pytest.mark.parametrize(
    "plain,original",
    [
        (PlainNSGAII, NSGA2Fair),
        (PlainNSGAIII, NSGAIIIPOFair),
        (PlainSPEA2, SPEA2POFair),
        (PlainRandomSearch, RandomSearchPOFair),
    ],
)
def test_method_specific_selectors_are_unchanged(plain, original):
    assert plain._environmental_selection is original._environmental_selection
    assert plain._tournament_select_one is original._tournament_select_one
    assert plain._get_objective_vectors is original._get_objective_vectors


@pytest.mark.parametrize("cls", PLAIN_BASELINES.values())
def test_plain_baselines_cannot_enable_custom_objective_editors(cls):
    with pytest.raises(ValueError, match="original CAPO variation"):
        optimizer(cls, objective_aware_variation=True)


def test_random_search_remains_independent_not_evolutionary_crossover():
    assert (
        PlainRandomSearch._generate_challengers
        is RandomSearchPOFair._generate_challengers
    )


def test_warmup_reaches_actual_readiness_before_evolution():
    obj = ready_optimizer()
    obj._pre_optimization_loop()
    assert len(obj.supported_initial_blocks) == 3
    assert obj._supported(obj.prompts, obj.supported_initial_blocks)
    assert obj.task.min_group_count == 6
    assert obj.task.fairness_kwargs["max_rate_ci_width"] == 1.0
    assert (
        obj.warmup_downstream_tokens > 0
        and obj.algorithm_counts["warmup_added_blocks"] == 2
    )
    assert len(obj.supported_initial_blocks) < obj.task.n_blocks


def test_warmup_uses_existing_metadata_selector(monkeypatch):
    obj = ready_optimizer()
    calls = []
    original = obj.task.select_fairness_block

    def choose(candidates, **kwargs):
        calls.append((list(candidates), kwargs))
        return original(candidates, **kwargs)

    monkeypatch.setattr(obj.task, "select_fairness_block", choose)
    obj._pre_optimization_loop()
    assert len(calls) == 2
    assert all("current_blocks" in k and "diagnostics" in k for _, k in calls)


def test_warmup_uses_search_rows_only():
    obj = ready_optimizer()
    ids_before = list(obj.task.ids)
    blocks_before = [list(b) for b in obj.task.blocks]
    obj._pre_optimization_loop()
    assert obj.task.ids == ids_before and obj.task.blocks == blocks_before
    assert all(identifier in obj.task.ids for _, identifier in obj.task.pred_cache)
    assert not obj.search_diagnostics()["holdout_feedback_used"]


def test_legacy_reproduction_remains_one_block_with_unsupported_evidence():
    obj = ready_optimizer(MOCAPOFairOptimizer)
    obj._pre_optimization_loop()
    assert len(obj.initial_common_blocks) == 1
    assert not obj._supported(obj.prompts, obj.initial_common_blocks)


def test_no_warmup_ablation_preserves_old_algorithm_trajectory():
    obj = ready_optimizer(MOCAPOFairOptimizer)
    obj.optimize(2)
    expected = [p.construct_prompt() for p in obj.prompts]
    other = ready_optimizer(bootstrap_readiness=False)
    other.optimize(2)
    assert [p.construct_prompt() for p in other.prompts] == expected


def test_impossible_support_fails_without_lowering_threshold():
    obj = ready_optimizer()
    obj.task.min_group_count = 1000
    with pytest.raises(RuntimeError, match="unchanged fairness-readiness"):
        obj._pre_optimization_loop()
    assert obj.task.min_group_count == 1000


def test_budget_guard_applies_during_warmup():
    reference = ready_optimizer(bootstrap_readiness=False)
    reference._pre_optimization_loop()
    obj = ready_optimizer()
    obj.budget_controller.max_tokens = reference.budget_controller.consumed_tokens + 1
    with pytest.raises(BudgetExhausted):
        obj._pre_optimization_loop()
    assert obj.budget_controller.consumed_tokens <= obj.budget_controller.max_tokens


def test_checkpoint_preserves_warmup_and_random_stream():
    obj = ready_optimizer()
    obj.optimize(1)
    state = pickle.loads(pickle.dumps(capture_optimizer_state(obj)))
    obj._step()
    expected = [p.construct_prompt() for p in obj.prompts]
    restored = ready_optimizer()
    restore_optimizer_state(restored, state)
    restored._step()
    assert [p.construct_prompt() for p in restored.prompts] == expected
    assert restored.supported_initial_blocks == obj.supported_initial_blocks
    assert restored.warmup_downstream_tokens == obj.warmup_downstream_tokens


def test_checkpoint_rejects_changed_readiness_schedule():
    obj = ready_optimizer()
    other = ready_optimizer(bootstrap_readiness=False)
    with pytest.raises(ValueError, match="initialization configuration"):
        other.load_optimizer_state_dict(obj.optimizer_state_dict())


@pytest.mark.parametrize("method", ["Tri-Fair-v9.3", *PLAIN_BASELINES])
def test_public_runner_and_frozen_audit(method, tmp_path, monkeypatch):
    from tests.test_integration import test_complete_public_runner_with_fake_models

    test_complete_public_runner_with_fake_models(
        method, tmp_path, monkeypatch, audit=True
    )


@pytest.mark.parametrize("profile,runs", [("ablation", 8), ("screen", 20)])
def test_launch_plan_distinguishes_protocols(monkeypatch, profile, runs):
    monkeypatch.setenv("V93_PROFILE", profile)
    monkeypatch.setenv("PILOT_SEEDS", "72")
    p = Path(__file__).resolve().parents[1] / "research/run_pilot_v93.py"
    spec = importlib.util.spec_from_file_location("plan93", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    plan = mod.plan()
    assert plan["optimizer_runs"] == runs and not plan["fairness_thresholds_changed"]
    if profile == "screen":
        assert set(PLAIN_BASELINES).issubset(plan["methods"])
    else:
        assert plan["methods"] == ("Tri-Fair-v9.3", "Tri-Fair-v9.2")


def test_reserve_reclamation_cannot_silently_change_new_protocol():
    from tri_fair.cli.train import _validate_args

    with pytest.raises(ValueError, match="frozen terminal audit"):
        _validate_args(
            SimpleNamespace(optimizer="Tri-Fair-v9.3", reclaim_audit_reserve=True)
        )


def test_random_search_plain_does_not_change_existing_random_algorithm():
    from tri_fair.optimizers.budget_fill import BudgetFilledRandomSearch

    random.seed(91)
    np.random.seed(91)
    old = optimizer(BudgetFilledRandomSearch)
    old.optimize(2)
    expected = [p.construct_prompt() for p in old.prompts]
    random.seed(91)
    np.random.seed(91)
    plain = optimizer(PlainRandomSearch)
    plain.optimize(2)
    assert [p.construct_prompt() for p in plain.prompts] == expected
    assert plain.predictor.llm.get_token_count() == old.predictor.llm.get_token_count()


def test_plain_initialization_is_shared_with_standard_full_evaluation():
    from tri_fair.optimizers.nsga2_core import NSGA2PromptOptimizer

    assert PlainNSGAII._pre_optimization_loop is NSGA2Fair._pre_optimization_loop
    assert NSGA2Fair.__mro__[2] is NSGA2PromptOptimizer


def test_plain_protocol_figures_preserve_new_run_identities(tmp_path, monkeypatch):
    from tests.test_research_figures import rows

    research = Path(__file__).resolve().parents[1] / "research"
    monkeypatch.syspath_prepend(str(research))
    import plot_v92 as base

    original = base.STYLE
    spec = importlib.util.spec_from_file_location("plot93", research / "plot_v93.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        frame = rows().replace(
            {
                "optimizer": {
                    "Tri-Fair-v8.6": "Tri-Fair-v9.3",
                    "NSGAII-PO-Fair": "NSGAII-PO-Fair-Plain",
                }
            }
        )
        frame["validation_input_tokens"] = 50.0
        frame["validation_output_tokens"] = 2.0
        dest = tmp_path / "heldout_validation/qwen-3-30b/winobias/seed72"
        dest.mkdir(parents=True)
        frame.to_parquet(dest / "eval_checkpoints_union.parquet", index=False)
        made = base.render(tmp_path, tmp_path / "figures", budget=1000)
        assert len(made) == 3 and not list((tmp_path / "figures").glob("*trajectory*"))
    finally:
        base.STYLE = original
