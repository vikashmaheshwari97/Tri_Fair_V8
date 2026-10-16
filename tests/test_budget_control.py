from __future__ import annotations

from dataclasses import dataclass

import pytest

from tri_fair.runtime.budget import (
    BudgetAwarePredictor,
    BudgetExhausted,
    TokenBudgetController,
)


class FakeTokenizer:
    def encode(self, text: str):
        # Deterministic tokenization suitable for controller unit tests.
        return str(text).split()


@dataclass
class FakeSamplingParams:
    max_tokens: int = 4


class FakeLLM:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.sampling_params = FakeSamplingParams()
        self.input_token_count = 0
        self.output_token_count = 0

    def get_token_count(self):
        return {
            "input_tokens": self.input_token_count,
            "output_tokens": self.output_token_count,
            "total_tokens": self.input_token_count + self.output_token_count,
        }


class FakePredictor:
    def __init__(self, llm: FakeLLM):
        self.llm = llm
        self.classes = ["a", "b"]
        self.extraction_description = "fake"

    def predict(self, prompts, xs, system_prompts=None):
        if system_prompts is None:
            systems = ["system"] * len(prompts)
        elif isinstance(system_prompts, str):
            systems = [system_prompts] * len(prompts)
        else:
            systems = list(system_prompts)
        inputs = [f"{prompt}\n{x}" for prompt, x in zip(prompts, xs)]
        outputs = ["a"] * len(inputs)
        self.llm.input_token_count += sum(
            len(self.llm.tokenizer.encode(value)) for value in inputs + systems
        )
        self.llm.output_token_count += sum(
            len(self.llm.tokenizer.encode(value)) for value in outputs
        )
        return outputs, [f"{x}\na" for x in xs]


def test_rejected_call_does_not_change_counters():
    llm = FakeLLM()
    controller = TokenBudgetController(
        llm=llm,
        max_tokens=5,
        max_output_tokens=4,
        output_token_reserve=0,
    )
    predictor = BudgetAwarePredictor(FakePredictor(llm), controller)

    with pytest.raises(BudgetExhausted):
        predictor.predict(["one two"], ["three four"])

    assert llm.get_token_count()["total_tokens"] == 0
    assert controller.rejected_operations == 1


def test_admitted_call_stays_below_hard_limit():
    llm = FakeLLM()
    controller = TokenBudgetController(
        llm=llm,
        max_tokens=20,
        max_output_tokens=4,
        output_token_reserve=1,
    )
    predictor = BudgetAwarePredictor(FakePredictor(llm), controller)

    preds, _ = predictor.predict(["one two"], ["three"])
    assert preds == ["a"]
    assert llm.get_token_count()["total_tokens"] <= 20
    assert controller.completed_downstream_calls == 1


def test_candidate_count_reduces_smoothly():
    llm = FakeLLM()
    controller = TokenBudgetController(
        llm=llm,
        max_tokens=100,
        max_output_tokens=4,
        near_budget_fraction=0.90,
    )
    assert controller.suggested_candidate_count(4) == 4

    llm.input_token_count = 95
    assert controller.suggested_candidate_count(4) == 2

    llm.input_token_count = 99
    assert controller.suggested_candidate_count(4) == 1


def test_state_round_trip():
    llm = FakeLLM()
    first = TokenBudgetController(
        llm=llm,
        max_tokens=100,
        max_output_tokens=4,
        near_budget_fraction=0.9,
        output_token_reserve=2,
    )
    first.request_stop(
        reason="next_complete_candidate_exceeds_budget",
        operation="full_candidate",
        estimated_tokens=20,
    )

    second = TokenBudgetController(
        llm=llm,
        max_tokens=100,
        max_output_tokens=4,
        near_budget_fraction=0.9,
        output_token_reserve=2,
    )
    second.load_state_dict(first.state_dict())
    assert second.stop_requested
    assert second.stopping_reason == "next_complete_candidate_exceeds_budget"
