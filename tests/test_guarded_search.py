"""Behavioral safeguards for the v8.7 redesign; no external LLM or dataset."""

import json
import random
import numpy as np
import pytest
from tests.fakes import optimizer, FakePredictor
from tri_fair.optimizers.guarded import (
    GuardedTriFairOptimizer,
    normalize_contract,
    CONTRACT,
)
from tri_fair.optimizers.nsga2_core import NSGA2PromptOptimizer
from tri_fair.runtime.budget import BudgetAwarePredictor, TokenBudgetController
from tri_fair.runtime.finalization import reclaim_and_finalize
from tri_fair.runtime.checkpointing import (
    capture_optimizer_state,
    restore_optimizer_state,
)


def test_output_contract_is_idempotent_and_keeps_conditional_label_mapping():
    original = "If the comment is toxic, use a; otherwise use b. Return exactly one label in <final_answer> tags. Output no explanation."
    clean = normalize_contract(original)
    assert clean.startswith("If the comment is toxic, use a; otherwise use b.")
    assert "Output accountant or journalist." in normalize_contract(
        "Output accountant or journalist.", ["accountant", "journalist"]
    )
    assert clean.count(CONTRACT) == 1
    assert normalize_contract(clean) == clean
    assert "Output a when the sentence describes abuse." in normalize_contract(
        "Output a when the sentence describes abuse. Return one label only."
    )


def test_complete_coverage_cached_guard_and_no_extra_inference():
    obj = optimizer(GuardedTriFairOptimizer, repeat_check_examples=4)
    obj.optimize(2)
    assert obj.repeatability_result["status"] == "passed"
    assert obj.repeatability_result["prediction_count"] == 12
    assert obj.repeatability_result["actual_downstream_tokens"] > 0
    assert (
        GuardedTriFairOptimizer._environmental_selection
        is NSGA2PromptOptimizer._environmental_selection
    )
    for p in obj.prompts:
        assert set(obj.task.get_evaluated_blocks([p])[p]) == set(
            range(obj.task.n_blocks)
        )
        before = obj.predictor.llm.get_token_count().copy()
        row = obj._guard_record(p)
        assert row["usable_folds"] == obj.task.n_blocks and row["supported"]
        assert obj.predictor.llm.get_token_count() == before
        assert row["quality"] == pytest.approx(obj.get_fairness_record(p)["quality"])


def test_backend_label_disagreements_fail_before_search_and_are_saved(tmp_path):
    class BatchDependent(FakePredictor):
        def predict(self, prompts, xs, system_prompts=None):
            labels, seq = super().predict(prompts, xs, system_prompts)
            if len(xs) == 1:
                labels = ["teacher" if y == "nurse" else "nurse" for y in labels]
            return labels, seq

    obj = optimizer(GuardedTriFairOptimizer, repeat_check_examples=4)
    obj.predictor = BatchDependent()
    obj.set_artifact_directory(tmp_path)
    with pytest.raises(RuntimeError, match="repeatability probe"):
        obj.optimize(3)
    saved = json.loads((tmp_path / "repeatability_probe.json").read_text())
    assert saved["status"] == "failed" and saved["mismatches"]
    assert not obj.task.pred_cache
    assert obj.generation_calls == 0


def test_invalid_generation_does_not_manufacture_a_budget_stop():
    obj = optimizer(GuardedTriFairOptimizer, repeat_check_examples=0)
    controller = TokenBudgetController(
        llm=obj.predictor.llm, max_tokens=500000, max_output_tokens=16
    )
    obj.predictor = BudgetAwarePredictor(obj.predictor, controller)
    obj.budget_controller = controller
    obj._pre_optimization_loop()
    obj._propose_challengers = lambda: []
    previous = list(obj.prompts)
    assert obj._step() == previous
    assert not controller.stop_requested and controller.stopping_reason is None


def test_fold_support_unknown_is_explicit_and_uses_pooled_parent_comparison():
    obj = optimizer(GuardedTriFairOptimizer, repeat_check_examples=0)
    obj._pre_optimization_loop()
    for p in obj.prompts:
        record = obj._guard_record(p)
        obj._guard_cache[p.construct_prompt()] = dict(
            record,
            supported=False,
            quality_range=None,
            fairness_range=None,
            usable_folds=0,
        )
    p, record = obj._parent("fairness", 0)
    assert record["selection"] == "pooled_unknown_sensitivity"
    assert record["quality_range"] is None


def test_checkpoint_roundtrip_retains_exact_next_proposal():
    random.seed(72)
    np.random.seed(72)
    obj = optimizer(GuardedTriFairOptimizer, repeat_check_examples=0)
    obj.optimize(2)
    state = capture_optimizer_state(obj)
    expected = obj._generate_challengers()
    other = optimizer(GuardedTriFairOptimizer, repeat_check_examples=0)
    restore_optimizer_state(other, state)
    observed = other._generate_challengers()
    assert [p.construct_prompt() for p in expected] == [
        p.construct_prompt() for p in observed
    ]
    mismatch = optimizer(
        GuardedTriFairOptimizer, repeat_check_examples=0, prompt_word_limit=90
    )
    with pytest.raises(ValueError, match="configuration mismatch"):
        mismatch.load_optimizer_state_dict(obj.optimizer_state_dict())


def test_new_method_reclaims_shared_reserve_without_exceeding_cap(tmp_path):
    obj = optimizer(GuardedTriFairOptimizer, repeat_check_examples=4)
    controller = TokenBudgetController(
        llm=obj.predictor.llm, max_tokens=35000, max_output_tokens=16
    )
    obj.predictor = BudgetAwarePredictor(obj.predictor, controller)
    obj.budget_controller = controller
    obj.optimize(100)
    final = reclaim_and_finalize(
        obj, tmp_path, count=4, total_budget=60000, max_steps=100
    )
    assert final["status"] == "complete"
    assert final["reserve_reclamation"]["continued"]
    assert final["reserve_reclamation"]["continued_search_tokens"] > 0
    assert controller.consumed_tokens <= 60000
    assert final["audit_downstream_tokens"] == 0
    assert (
        obj.repeatability_result["actual_downstream_tokens"]
        < controller.consumed_tokens
    )


def test_five_method_commands_have_same_budget_and_protocol(capsys):
    from tri_fair.cli.pilot_v87 import main

    assert main(["--dataset", "civil_comments"]) == 0
    lines = [
        s for s in capsys.readouterr().out.splitlines() if "tri_fair.cli.train" in s
    ]
    assert len(lines) == 5
    for line in lines:
        assert "--budget-per-run 2000000" in line and "--reclaim-audit-reserve" in line
        assert "--n-init-prompts 4" in line and "--max-few-shot-examples 2" in line
    assert any("--optimizer NSGAII-PO-Fair " in s for s in lines)


def test_probe_rejects_consistently_invalid_labels_before_search(tmp_path):
    obj = optimizer(GuardedTriFairOptimizer, repeat_check_examples=4)
    original = obj.predictor.predict

    def invalid(prompts, xs, system_prompts=None):
        _, sequences = original(prompts, xs, system_prompts)
        return ["invalid"] * len(xs), sequences

    obj.predictor.predict = invalid
    obj.set_artifact_directory(tmp_path)
    with pytest.raises(RuntimeError, match="invalid labels"):
        obj.optimize(1)
    result = json.loads((tmp_path / "repeatability_probe.json").read_text())
    assert result["invalid_label_examples"] and not result["mismatches"]
    assert not obj.task.pred_cache


def test_revision_uses_measured_equalized_odds_diagnostics_without_judge_calls():
    obj = optimizer(GuardedTriFairOptimizer, repeat_check_examples=0)
    parent = obj.prompts[0]
    obj.task.fairness_metric_name = "civil_equalized_odds"
    obj.get_fairness_record = lambda p: dict(
        quality=0.8,
        cost=20,
        fairness=0.2,
        diagnostics=dict(
            metric="civil_equalized_odds",
            rates_by_identity={
                "muslim": dict(equalized_odds_gap=0.25, tpr_gap=0.1, fpr_gap=0.3)
            },
        ),
    )
    before = obj.meta_llm.get_token_count().copy()
    request = obj._request(
        parent,
        "fairness",
        {"evidence": [], "confusions": [], "conditional_error_clusters": []},
    )
    assert "worst_identities" in request and "muslim" in request
    assert "fpr_gap" in request and "0.3" in request
    assert "false positives and false negatives separately" in request
    assert obj.meta_llm.get_token_count() == before
