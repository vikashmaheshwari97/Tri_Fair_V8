"""Regression coverage for observed weaknesses; fake LLMs test mechanics only."""

import json
import random
import numpy as np
import pandas as pd
import pytest
from promptolution.utils.prompt import Prompt
from tests.fakes import optimizer
from tri_fair.optimizers.efficient import EfficientTriFairOptimizer, EFFICIENT_ARMS
from tri_fair.optimizers.baselines import NSGAIIIPOFair, SPEA2POFair, RandomSearchPOFair
from tri_fair.optimizers.nsga2 import NSGA2Fair
from tri_fair.runtime.budget import BudgetAwarePredictor, TokenBudgetController
from tri_fair.runtime.finalization import reclaim_and_finalize
from tri_fair.search.stability import search_stability, choose_stable_parent
from tri_fair.search.attribution import attribute_search


def initialized():
    obj = optimizer(
        EfficientTriFairOptimizer, smart_start_count=0, uncertainty_resamples=8
    )
    obj._pre_optimization_loop()
    return obj


def test_unknown_stability_remains_unknown_without_extra_predictions():
    obj = initialized()
    p = obj.incumbents[0]
    blocks = obj.task.get_evaluated_blocks([p])[p][:2]
    before = obj.predictor.llm.get_token_count().copy()
    row = search_stability(obj, p, blocks)
    assert row["usable_folds"] == 0
    assert row["sensitivity_supported"] is False
    assert row["conservative_quality"] is None and row["fairness_sensitivity"] is None
    assert obj.predictor.llm.get_token_count() == before
    obj.incumbents = [p]
    _, selected = choose_stable_parent(obj, "quality", 0)
    if not selected["sensitivity_supported"]:
        assert selected["selection_basis"] == "pooled_fallback_unknown_sensitivity"
        assert selected["selection_quality"] == selected["quality"]


def test_targeted_feedback_replaces_at_most_one_error_and_retains_success():
    obj = initialized()
    p = obj.incumbents[0]
    blocks = obj.task.get_evaluated_blocks([p])[p]
    before = obj.predictor.llm.get_token_count().copy()
    a = attribute_search(obj.task, p, blocks, limit=4)
    b = attribute_search(obj.task, p, blocks, limit=4, targeted=True)
    assert len(a["evidence"]) == len(b["evidence"])
    assert sum(x != y for x, y in zip(a["evidence"], b["evidence"])) <= 1
    for e in a["evidence"]:
        if e["gold"] == e["prediction"]:
            assert e in b["evidence"]
    assert len(json.dumps(b["evidence"], ensure_ascii=False)) <= 2600
    assert obj.predictor.llm.get_token_count() == before


def test_compaction_proposes_real_shorter_candidate_without_scoring_it():
    obj = initialized()
    p = Prompt(
        "Classify occupation using explicit job duties.",
        ["long example one", "long example two"],
    )
    obj._get_block_vectors([p], [0, 1, 2])
    obj.incumbents = [p]
    before = obj.predictor.llm.get_token_count().copy()
    meta = obj.meta_llm.get_token_count().copy()
    children = obj._compact_challengers()
    assert children and children[0].instruction == p.instruction
    assert len(children[0].few_shots) == 1
    assert obj.predictor.llm.get_token_count() == before
    assert obj.meta_llm.get_token_count() == meta
    assert not obj.task.get_evaluated_blocks(children)[children[0]]


def test_merge_gate_blocks_missing_pair_and_enforces_cooldown():
    obj = initialized()
    obj.incumbents = obj.incumbents[:1]
    assert obj._merge_eligible() == (False, "no_observed_pair")
    obj.last_merge_round = sum(obj.portfolio_trials.values())
    assert obj._merge_eligible() == (False, "cooldown")


def test_efficiency_state_roundtrip_and_registry_enables_compaction():
    from tri_fair.config.study import build_optimizer_registry

    random.seed(72)
    np.random.seed(72)
    obj = optimizer(
        EfficientTriFairOptimizer, smart_start_count=0, uncertainty_resamples=8
    )
    obj.optimize(6)
    assert obj.portfolio_trials["compact_edit"] > 0
    assert obj.search_diagnostics()["supported_stability_required"]
    other = optimizer(
        EfficientTriFairOptimizer, smart_start_count=0, uncertainty_resamples=8
    )
    other.load_optimizer_state_dict(obj.optimizer_state_dict())
    assert other.portfolio_trials == obj.portfolio_trials
    assert other.compact_attempts == obj.compact_attempts
    assert set(
        build_optimizer_registry()["Tri-Fair-v8.6"].optimizer_params[
            "enabled_strategies"
        ]
    ) == set(EFFICIENT_ARMS)


@pytest.mark.parametrize(
    "cls",
    [
        EfficientTriFairOptimizer,
        NSGA2Fair,
        NSGAIIIPOFair,
        SPEA2POFair,
        RandomSearchPOFair,
    ],
)
def test_reserve_reclamation_same_cap_complete_finalists_and_accounting(tmp_path, cls):
    random.seed(72)
    np.random.seed(72)
    kwargs = (
        dict(smart_start_count=0, uncertainty_resamples=8)
        if cls is EfficientTriFairOptimizer
        else {}
    )
    obj = optimizer(cls, **kwargs)
    controller = TokenBudgetController(
        llm=obj.predictor.llm, max_tokens=35000, max_output_tokens=16
    )
    obj.predictor = BudgetAwarePredictor(obj.predictor, controller)
    obj.budget_controller = controller
    obj.optimize(100)
    before = controller.consumed_tokens
    initial_steps = obj._completed_steps
    result = reclaim_and_finalize(
        obj, tmp_path, count=4, total_budget=60000, max_steps=100
    )
    assert result["status"] == "complete"
    assert controller.consumed_tokens <= 60000
    phase = result["reserve_reclamation"]
    assert (
        sum(
            phase[x]
            for x in [
                "initial_search_tokens",
                "coverage_tokens",
                "continued_search_tokens",
            ]
        )
        == controller.consumed_tokens
    )
    assert phase["continued"] and phase["continued_search_tokens"] > 0
    assert obj._completed_steps >= initial_steps
    assert controller.consumed_tokens > before
    assert result["audit_downstream_tokens"] == 0
    frame = pd.read_parquet(tmp_path / "final_search_results.parquet")
    assert frame.fairness_ready.all()
    assert all(
        set(json.loads(v)) == set(range(obj.task.n_blocks))
        for v in frame.evaluated_on_blocks
    )
    assert 0 < len(frame) <= 4


def test_one_seed_four_methods_submission_and_shared_reclaim_flag(capsys):
    from tri_fair.cli.pilot_v86 import main

    methods = "Tri-Fair-v8.6,NSGAIII-PO-Fair,SPEA2-PO-Fair,RandomSearch-PO-Fair"
    assert (
        main(["--dataset", "winobias", "--methods", methods, "--reclaim-audit-reserve"])
        == 0
    )
    lines = [
        v for v in capsys.readouterr().out.splitlines() if "tri_fair.cli.train" in v
    ]
    assert len(lines) == 4 and all("--reclaim-audit-reserve" in v for v in lines)
