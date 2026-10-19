"""One shared optimizer, actual search credit and reproducible adaptation."""

import random
import numpy as np
import pytest
from tests.fakes import optimizer
from tri_fair.optimizers.consolidated import ConsolidatedTriFairOptimizer
from tri_fair.search.portfolio import ARMS, select_arm, cached_front_gain


def test_scheduler_cold_start_and_reward_driven_selection():
    assert (
        select_arm(
            dict.fromkeys(ARMS, 0), dict.fromkeys(ARMS, 0), exploration=0.2, interval=5
        )
        == ARMS[0]
    )
    trials = dict.fromkeys(ARMS, 2)
    rewards = dict.fromkeys(ARMS, 0.0)
    rewards[ARMS[1]] = 1
    assert select_arm(trials, rewards, exploration=0, interval=5) == ARMS[1]
    trials[ARMS[0]] = 1
    trials[ARMS[1]] = 2
    trials[ARMS[2]] = 2
    assert select_arm(trials, rewards, exploration=0, interval=5) == ARMS[0]


def test_credit_never_calls_model_or_compares_different_blocks():
    obj = optimizer(
        ConsolidatedTriFairOptimizer, smart_start_count=0, uncertainty_resamples=8
    )
    obj._pre_optimization_loop()
    p = obj.incumbents[0]
    before = obj.predictor.llm.get_token_count().copy()
    result = cached_front_gain(obj, [p], [p])
    assert obj.predictor.llm.get_token_count() == before
    assert not result["eligible"] or result["hv_gain"] == 0


def test_all_three_strategies_share_one_archive_and_record_real_credit():
    random.seed(72)
    np.random.seed(72)
    obj = optimizer(
        ConsolidatedTriFairOptimizer, smart_start_count=0, uncertainty_resamples=8
    )
    obj.optimize(8)
    assert all(obj.portfolio_trials[a] > 0 for a in ARMS)
    assert len(obj.portfolio_trace) > 0
    assert all(0 <= r["reward"] <= 1 for r in obj.portfolio_trace)
    assert all(r["downstream_tokens_spent"] >= 0 for r in obj.portfolio_trace)
    assert all(r["eligible"] is False or r["blocks"] for r in obj.portfolio_trace)
    assert obj.search_diagnostics()["shared_budget"]
    state = obj.optimizer_state_dict()
    other = optimizer(
        ConsolidatedTriFairOptimizer, smart_start_count=0, uncertainty_resamples=8
    )
    other.load_optimizer_state_dict(state)
    assert other.portfolio_trials == obj.portfolio_trials
    assert other.portfolio_trace == obj.portfolio_trace
    mismatch = optimizer(ConsolidatedTriFairOptimizer, portfolio_exploration=0.5)
    with pytest.raises(ValueError, match="Portfolio checkpoint"):
        mismatch.load_optimizer_state_dict(state)


def test_only_one_trifair_in_active_configuration():
    from tri_fair.config.registry import ACTIVE_METHODS

    assert [x for x in ACTIVE_METHODS if x.startswith("Tri-Fair")] == ["Tri-Fair-v8.8"]


def test_component_ablation_selects_only_enabled_strategy():
    obj = optimizer(
        ConsolidatedTriFairOptimizer,
        smart_start_count=0,
        uncertainty_resamples=8,
        enabled_strategies=["attributed_reflection"],
    )
    obj.optimize(3)
    assert set(obj.portfolio_trials) == {"attributed_reflection"}
    assert obj.portfolio_trials["attributed_reflection"] > 0
    with pytest.raises(ValueError, match="proposal strategies"):
        optimizer(ConsolidatedTriFairOptimizer, enabled_strategies=["unknown"])


def test_no_false_budget_exhaustion_when_merge_has_no_second_parent():
    obj = optimizer(
        ConsolidatedTriFairOptimizer, smart_start_count=0, uncertainty_resamples=8
    )
    obj._pre_optimization_loop()
    obj.incumbents = obj.incumbents[:1]
    assert obj._merge_challengers() == []
    assert obj.proposal_rejections["merge_requires_two_observed_parents"] == 1

