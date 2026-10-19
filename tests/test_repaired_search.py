"""Regression coverage for the actual v8.7 rendered-demo failure."""

import json
import random
import numpy as np
import pytest
from promptolution.utils.prompt import Prompt
from tests.fakes import optimizer, FakeLLM, FakePredictor
from tri_fair.optimizers.repaired import RepairedTriFairOptimizer
from tri_fair.optimizers.nsga2_core import NSGA2PromptOptimizer
from tri_fair.search.rendering import rendered_prompt, output_contract
from tri_fair.runtime.checkpointing import (
    capture_optimizer_state,
    restore_optimizer_state,
)
from tri_fair.runtime.budget import BudgetAwarePredictor, TokenBudgetController
from tri_fair.runtime.finalization import reclaim_and_finalize


def test_reproduces_historical_bug_then_verifies_actual_model_input():
    p = Prompt("Classify the input.")
    demo = "Input:\nDEMO EVIDENCE\nOutput:\n<final_answer>a</final_answer>"
    old = Prompt(p.instruction, [demo], downstream_template=p.downstream_template)
    assert demo not in old.construct_prompt()
    new = rendered_prompt(p.instruction, [demo], template=p.downstream_template)
    assert demo in new.construct_prompt()
    assert new.construct_prompt() != p.construct_prompt()
    assert (
        rendered_prompt(
            p.instruction, [], template=new.downstream_template
        ).construct_prompt()
        == p.construct_prompt()
    )


def test_custom_scaffold_preserved_and_output_rules_not_duplicated():
    demo = "Input:\ncase\nOutput:\nlabel"
    p = rendered_prompt("rule", [demo], template="HEADER\n<instruction>\nFOOTER")
    assert p.construct_prompt().startswith("HEADER")
    assert p.construct_prompt().endswith("FOOTER")
    assert demo in p.construct_prompt()
    instruction = "If abusive, return toxic; otherwise non_toxic. Return toxic or non_toxic in <final_answer> tags."
    clean = output_contract(instruction)
    assert clean == instruction and output_contract(clean) == clean
    assert (
        output_contract(
            clean + " Return one allowed label in <final_answer> tags; no explanation."
        )
        == clean
    )


def test_initialization_and_demo_edits_reach_predictor_not_only_metadata():
    class Recording(FakePredictor):
        def __init__(self):
            super().__init__()
            self.inputs = []

        def predict(self, prompts, xs, system_prompts=None):
            self.inputs.extend(prompts)
            return super().predict(prompts, xs, system_prompts)

    obj = optimizer(RepairedTriFairOptimizer, repeat_check_examples=0)
    obj.predictor = Recording()
    obj._pre_optimization_loop()
    with_demos = [p for p in obj.prompts if p.few_shots]
    assert with_demos
    for p in with_demos:
        assert p.construct_prompt() in obj.predictor.inputs
        assert all(d in p.construct_prompt() for d in p.few_shots)
    obj.proposal_round = 1  # demonstration mode
    children = obj._propose_challengers()
    assert children
    assert any(x[2] == "demonstrations" for x in obj._pending_edits)
    assert all(all(d in p.construct_prompt() for d in p.few_shots) for p in children)


def test_zero_shot_cost_mode_calls_compressor_and_requires_real_token_reduction():
    class Compressor(FakeLLM):
        def get_response(self, prompts, system_prompts=None):
            self.input_token_count += sum(len(p.split()) for p in prompts)
            self.output_token_count += 8 * len(prompts)
            return [
                json.dumps(
                    dict(
                        diagnosis="remove repetition",
                        instruction="Classify occupation.",
                    )
                )
                for _ in prompts
            ]

    obj = optimizer(RepairedTriFairOptimizer, repeat_check_examples=0)
    obj._pre_optimization_loop()
    parent = next(p for p in obj.prompts if not p.few_shots)
    obj._parent = lambda mode, offset: (parent, obj._guard_record(parent))
    obj.meta_llm = Compressor()
    obj.crossovers_per_iter = 1
    obj.proposal_round = 4
    children = obj._propose_challengers()
    assert children and obj.generation_calls == 1
    assert obj.token_counter(children[0].construct_prompt()) < obj.token_counter(
        parent.construct_prompt()
    )


def test_archive_never_requests_uncached_predictions_and_survival_is_unchanged():
    obj = optimizer(
        RepairedTriFairOptimizer, repeat_check_examples=0, parent_pool_cap=4
    )
    obj.optimize(3)
    before = obj.predictor.llm.get_token_count().copy()
    obj._update_archive(obj.prompts)
    for mode in ("quality", "cost", "fairness", "balanced", "explore"):
        obj._parent(mode, 1)
    assert obj.predictor.llm.get_token_count() == before
    assert len(obj.candidate_archive) <= 4
    assert (
        RepairedTriFairOptimizer._environmental_selection
        is NSGA2PromptOptimizer._environmental_selection
    )
    assert all(
        set(obj.task.get_evaluated_blocks([p])[p]) == set(range(obj.task.n_blocks))
        for p in obj.candidate_archive
    )


def test_feedback_contains_measured_outcomes_not_unverified_success_claims():
    obj = optimizer(RepairedTriFairOptimizer, repeat_check_examples=0)
    obj.optimize(2)
    assert obj.edit_memory and sum(obj.operator_evaluations.values()) > 0
    for record in obj.edit_memory:
        assert all(
            np.isfinite(record[k])
            for k in ("quality_change", "cost_change", "unfairness_change")
        )
    assert obj.search_diagnostics()["holdout_feedback_used"] is False


def test_checkpoint_preserves_archive_and_next_proposal():
    random.seed(72)
    np.random.seed(72)
    obj = optimizer(RepairedTriFairOptimizer, repeat_check_examples=0)
    obj.optimize(2)
    state = capture_optimizer_state(obj)
    expected = obj._propose_challengers()
    other = optimizer(RepairedTriFairOptimizer, repeat_check_examples=0)
    restore_optimizer_state(other, state)
    actual = other._propose_challengers()
    assert [p.construct_prompt() for p in actual] == [
        p.construct_prompt() for p in expected
    ]
    assert other.edit_memory == obj.edit_memory
    mismatch = optimizer(
        RepairedTriFairOptimizer, repeat_check_examples=0, parent_pool_cap=12
    )
    with pytest.raises(ValueError, match="configuration mismatch"):
        mismatch.load_optimizer_state_dict(state["optimizer_extra"])


def test_budget_reclamation_counts_rendered_demos_and_respects_cap(tmp_path):
    obj = optimizer(RepairedTriFairOptimizer, repeat_check_examples=4)
    controller = TokenBudgetController(
        llm=obj.predictor.llm, max_tokens=35000, max_output_tokens=16
    )
    obj.predictor = BudgetAwarePredictor(obj.predictor, controller)
    obj.budget_controller = controller
    obj.optimize(100)
    result = reclaim_and_finalize(
        obj, tmp_path, count=4, total_budget=60000, max_steps=100
    )
    assert result["status"] == "complete"
    assert controller.consumed_tokens <= 60000
    assert obj.repeatability_result["actual_downstream_tokens"] > 0
    assert result["audit_downstream_tokens"] == 0


def test_five_method_protocol_keeps_matched_settings_and_independent_namespaces(capsys):
    from tri_fair.cli.pilot_v88 import main

    assert main(["--dataset", "civil_comments"]) == 0
    lines = [
        s for s in capsys.readouterr().out.splitlines() if "tri_fair.cli.train" in s
    ]
    assert len(lines) == 5
    for line in lines:
        assert "--budget-per-run 2000000" in line
        assert "--n-init-prompts 4" in line
        assert "--max-few-shot-examples 2" in line
        assert "--final-search-candidates 4" in line
        assert "--reclaim-audit-reserve" in line


def test_cpu_preflight_protects_baseline_and_rendering():
    from tri_fair.cli.preflight_v88 import check

    assert check()["ok"] is True


@pytest.mark.parametrize(
    "instruction,rejected",
    [
        ("Always return nurse.", True),
        ("Always return a valid label.", False),
        ("Return nurse when the duties are nursing; otherwise return teacher.", False),
    ],
)
def test_collapse_filter_keeps_generic_formatting_and_conditional_rules(
    instruction, rejected
):
    class Answer(FakeLLM):
        def get_response(self, prompts, system_prompts=None):
            return [json.dumps(dict(instruction=instruction)) for _ in prompts]

    obj = optimizer(RepairedTriFairOptimizer, repeat_check_examples=0)
    obj.meta_llm = Answer()
    obj._request = lambda *args: "test"
    obj.task.classes = ["a", "b", "nurse", "teacher"]
    _, _, reason = obj._generate_instruction(obj.prompts[0], "quality", {})
    assert (reason == "unconditional_class_collapse") is rejected


def test_resume_restores_parent_no_longer_in_live_population_without_inference():
    obj = optimizer(RepairedTriFairOptimizer, repeat_check_examples=0)
    obj._pre_optimization_loop()
    archived = rendered_prompt("An archived complete decision rule.")
    obj.task.evaluate([archived], obj.predictor, eval_strategy="full")
    obj.candidate_archive.append(archived)
    state = capture_optimizer_state(obj)
    other = optimizer(RepairedTriFairOptimizer, repeat_check_examples=0)
    restore_optimizer_state(other, state)
    before = other.predictor.llm.get_token_count().copy()
    assert set(other.task.get_evaluated_blocks([archived])[archived]) == set(
        range(other.task.n_blocks)
    )
    assert np.isfinite(other._guard_record(archived)["quality"])
    assert before == other.predictor.llm.get_token_count()


def test_winobias_demonstration_pair_is_rendered_and_removed_as_one_unit(tmp_path):
    from tri_fair.config.datasets import ALL_DATASETS
    from tri_fair.data.loading import create_dev_tasks
    from tri_fair.search.stability import balanced_gold_demonstrations

    task, pool = create_dev_tasks(
        ALL_DATASETS["winobias"], "full", seed=72, manifest_dir=tmp_path
    )
    obj = optimizer(RepairedTriFairOptimizer, repeat_check_examples=0)
    obj.task, obj.df_few_shots = task, pool
    demos = balanced_gold_demonstrations(obj, {}, offset=0)
    assert len(demos) == 2
    rows = pool[pool.input.map(lambda x: any(x in d for d in demos))]
    assert rows.pair_id.nunique() == 1 and rows.stereotype.nunique() == 2
    prompt = rendered_prompt("Resolve a or b.", demos, template="<instruction>")
    assert all(d in prompt.construct_prompt() for d in demos)
    obj._parent = lambda mode, offset: (prompt, {})
    obj._guard_record = lambda p: {}
    obj.task.cached_prediction_frame = lambda p, b: __import__("pandas").DataFrame(
        columns=["_review_target", "_review_prediction", "pair_id"]
    )
    obj.proposal_round = 4
    obj.crossovers_per_iter = 1
    child = obj._propose_challengers()[0]
    assert not child.few_shots
    assert all(d not in child.construct_prompt() for d in demos)
