"""Regression for actual Bios/WinoBias stops with estimated=0 and spare budget."""

import json
import pytest
from promptolution.utils.prompt import Prompt
from tests.fakes import optimizer, FakeLLM
from tri_fair.optimizers.hybrid import HybridTriFairOptimizer
from tri_fair.runtime.budget import (
    TokenBudgetController,
    BudgetAwarePredictor,
    BudgetExhausted,
)


def initialized():
    obj = optimizer(HybridTriFairOptimizer)
    obj.upper_shots = 0
    obj._pre_optimization_loop()
    llm = obj.predictor.llm
    controller = TokenBudgetController(
        llm=llm, max_tokens=1000000, max_output_tokens=16
    )
    obj.predictor = BudgetAwarePredictor(obj.predictor, controller)
    obj.budget_controller = controller
    obj._resume_loaded = True
    obj._completed_steps = 0
    return obj, controller


def test_observed_rejected_compression_does_not_stop_search(monkeypatch):
    obj, controller = initialized()
    # Force the actual rejected-compression branch; keep demonstrations absent
    # so deletion cannot replace the deliberately oversized rewrite.

    class LongerRewrite(FakeLLM):
        def get_response(self, prompts, system_prompts=None):
            return [
                json.dumps(
                    {
                        "instruction": " ".join(
                            ["Use explicit professional evidence."] * 10
                        )
                    }
                )
                for _ in prompts
            ]

    obj.meta_llm = LongerRewrite()
    obj.portfolio_round = 1
    before = controller.consumed_tokens
    obj._step()
    assert obj.proposal_rejections["compression_not_shorter"] == 1
    assert obj.empty_generation_streak == 1
    assert not controller.stop_requested and controller.stopping_reason is None
    assert controller.consumed_tokens == before
    assert controller.remaining_tokens > 0
    # The next call must be allowed to reach a later portfolio operator.
    novel = Prompt("Use roles and credentials. Distinct next-operator candidate.", [])
    monkeypatch.setattr(obj, "_propose_challengers", lambda: [novel])
    obj._step()
    assert controller.consumed_tokens > before
    assert obj.empty_generation_streak == 0
    assert not controller.stop_requested


def test_three_empty_proposals_stop_honestly_without_infinite_loop(monkeypatch):
    obj, controller = initialized()
    monkeypatch.setattr(obj, "_propose_challengers", lambda: [])
    before = controller.consumed_tokens
    obj.optimize(50)
    assert obj._completed_steps == 3
    assert obj.search_stop_reason == "no_novel_candidates_after_three_rounds"
    assert not controller.stop_requested and controller.stopping_reason is None
    assert controller.consumed_tokens == before


def test_real_oversized_candidate_still_raises_budget_exhaustion(monkeypatch):
    obj, controller = initialized()
    controller.max_tokens = controller.consumed_tokens + 1
    monkeypatch.setattr(
        obj,
        "_propose_challengers",
        lambda: [Prompt("A new candidate requiring full search prediction.", [])],
    )
    with pytest.raises(BudgetExhausted) as caught:
        obj._step()
    assert caught.value.estimated_tokens > 0
    assert controller.stop_requested
    assert controller.consumed_tokens <= controller.max_tokens


def test_generation_limit_is_not_relabelled_as_budget_exhaustion():
    obj, controller = initialized()
    obj.generation_calls = obj.max_generation_calls
    obj.optimize(50)
    assert obj._completed_steps == 1
    assert obj.search_stop_reason == "generation_call_limit"
    assert not controller.stop_requested


def test_empty_proposal_state_is_checkpointed():
    obj, controller = initialized()
    obj.empty_generation_streak = 2
    state = obj.optimizer_state_dict()
    other = optimizer(HybridTriFairOptimizer)
    other.load_optimizer_state_dict(state)
    assert other.empty_generation_streak == 2
