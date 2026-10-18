"""Regressions for the real stagnant-proposal log and bounded attributed search."""

from argparse import Namespace
import json
import pytest
from tests.fakes import FakeLLM, optimizer
from tri_fair.optimizers.attributed import AttributedTriFairOptimizer
from tri_fair.search.attribution import (
    attribute_search,
    gold_demonstration,
    parse_proposal,
    revision_request,
)
from tri_fair.runtime.progress import EvaluationProgressCallback
from tri_fair.cli.pilot import train_command


@pytest.mark.parametrize(
    "response,reason",
    [
        (
            "<prompt>Return <final_answer>nurse</final_answer>.</prompt>",
            "hardcoded_class_answer",
        ),
        ('{"instruction":null}', "missing_instruction"),
        ("<prompt>Input: biography Output: nurse</prompt>", "embedded_example_or_code"),
        ("no structured proposal", "missing_instruction"),
    ],
)
def test_rejects_observed_literal_answers_and_malformed_proposals(response, reason):
    instruction, _, error = parse_proposal(response, ["nurse", "teacher"])
    assert not instruction and error == reason


def test_json_diagnosis_and_complete_operational_instruction():
    text, diagnosis, reason = parse_proposal(
        json.dumps(
            {
                "diagnosis": "confuses title with hobby",
                "instruction": "Use explicit current professional duties; ignore hobbies.",
            }
        ),
        ["nurse", "teacher"],
    )
    assert (
        reason is None
        and diagnosis == "confuses title with hobby"
        and text.startswith("Use explicit")
    )


class MalformedLLM(FakeLLM):
    def __init__(self):
        super().__init__()
        self.requests = []

    def get_response(self, prompts, system_prompts=None):
        self.requests.extend(prompts)
        self.input_token_count += sum(len(p.split()) for p in prompts)
        self.output_token_count += len(prompts)
        return ["invalid" for _ in prompts]


def fully_initialized():
    obj = optimizer(
        AttributedTriFairOptimizer,
        smart_start_count=0,
        uncertainty_resamples=8,
        max_idle_rounds=2,
    )
    obj._pre_optimization_loop()
    obj._get_block_vectors(obj.prompts, list(range(obj.task.n_blocks)))
    obj._resume_loaded = True
    obj._completed_steps = 0
    return obj


def test_invalid_duplicates_stop_after_two_idle_rounds_instead_of_746_steps():
    obj = fully_initialized()
    bad = MalformedLLM()
    obj.meta_llm = bad
    before = obj.predictor.llm.get_token_count().copy()
    obj.optimize(100)
    assert obj._completed_steps == 2
    assert obj.generation_calls == 2
    assert obj.search_stop_reason == "no_new_evaluations"
    assert obj.predictor.llm.get_token_count() == before
    assert sum(obj.mode_attempts.values()) >= 2
    assert len([mode for mode, n in obj.mode_attempts.items() if n]) >= 2
    assert bad.requests[0] != bad.requests[-1]
    assert obj.proposal_rejections["missing_instruction"] >= 2


def test_generation_call_limit_stops_without_another_llm_call():
    obj = fully_initialized()
    obj.generation_calls = obj.max_generation_calls
    before = obj.meta_llm.get_token_count()
    assert obj._generate_challengers() == []
    assert obj.meta_llm.get_token_count() == before
    assert obj.search_stop_reason == "generation_call_limit"


def test_valid_but_repeated_proposals_are_counted_and_stopped():
    class RepeatingLLM(MalformedLLM):
        def get_response(self, prompts, system_prompts=None):
            super().get_response(prompts, system_prompts)
            return [
                json.dumps(
                    {
                        "instruction": "Classify the occupation using current principal duties."
                    }
                )
                for _ in prompts
            ]

    obj = fully_initialized()
    obj.meta_llm = RepeatingLLM()
    obj.enable_demo_search = False
    # Mark both catalog and compact variants as previously tried, reproducing
    # the old generation/deduplication path without paying for new evaluations.
    from tri_fair.search.attribution import contract_instruction
    from promptolution.utils.prompt import Prompt

    for catalog in (False, True):
        obj.seen_candidates.add(
            Prompt(
                contract_instruction(
                    "Classify the occupation using current principal duties.",
                    obj.task.classes,
                    catalog=catalog,
                ),
                [],
            ).construct_prompt()
        )
    obj.optimize(100)
    assert obj._completed_steps == 2
    assert obj.proposal_rejections["duplicate_candidate"] >= 2
    assert len([mode for mode, n in obj.mode_attempts.items() if n]) >= 2


def test_attribution_detects_raw_parser_fallback_without_changing_scores():
    obj = fully_initialized()
    prompt = obj.incumbents[0]
    key = next(
        key for key in obj.task.sequence_cache if key[0] == prompt.construct_prompt()
    )
    old = obj.task.pred_cache[key]
    obj.task.sequence_cache[key] = "Output without required tags"
    attribution = attribute_search(obj.task, prompt, list(range(obj.task.n_blocks)))
    assert attribution["invalid_output_count"] == 1
    assert obj.task.pred_cache[key] == old
    assert attribution["conditional_error_clusters"]


def test_frozen_fewshot_gold_labels_need_no_generation_and_no_search_examples():
    obj = fully_initialized()
    before = obj.predictor.llm.get_token_count()
    demo = gold_demonstration(obj, target="teacher")
    assert demo and "<final_answer>teacher</final_answer>" in demo
    assert obj.predictor.llm.get_token_count() == before
    assert set(obj.df_few_shots._manifest_id).isdisjoint(obj.task.df._manifest_id)


def test_context_limit_preserves_schema_without_touching_holdout():
    obj = fully_initialized()
    prompt = obj.incumbents[0]
    attribution = attribute_search(obj.task, prompt, list(range(obj.task.n_blocks)))
    obj.proposal_memory = [{"instruction": "long " * 300} for _ in range(6)]
    request = revision_request(obj, prompt, prompt, "fairness", attribution, attempt=3)
    assert (
        len(obj.meta_llm.tokenizer.encode(request)) <= obj.generation_input_token_limit
    )
    assert '"allowed_labels": ["nurse", "teacher"]' in request
    assert '"parent_measurement"' in request


def test_shared_progress_watchdog_allows_flat_scores_with_fresh_evaluations():
    obj = fully_initialized()
    callback = EvaluationProgressCallback(2)
    assert callback.on_step_end(obj)
    assert callback.on_step_end(obj)
    obj.predictor.llm.input_token_count += 1
    assert callback.on_step_end(obj)
    assert callback.on_step_end(obj)
    assert not callback.on_step_end(obj)
    assert obj.runtime_stop_reason == "no_new_downstream_evaluations"


def test_shared_choice_grammar_and_resource_limits_apply_to_every_method(tmp_path):
    args = Namespace(
        model="qwen-3-30b",
        dataset="bias_in_bios",
        seed=72,
        budget=1000000,
        manifest_dir=str(tmp_path),
    )
    for method in ("Tri-Fair-v8.3", "NSGAII-PO-Fair", "SPEA2-PO-Fair"):
        command = train_command(args, method, tmp_path)
        assert "--structured-labels" in command
        assert command[command.index("--max-idle-steps") + 1] == "3"
        assert command[command.index("--meta-max-output-tokens") + 1] == "512"


def test_proposal_memory_contains_measured_values_and_checkpoint_identity():
    obj = optimizer(
        AttributedTriFairOptimizer, smart_start_count=0, uncertainty_resamples=8
    )
    obj.optimize(2)
    measured = [r for r in obj.proposal_memory if r.get("measurements")]
    assert measured
    assert all(m["blocks"] for r in measured for m in r["measurements"])
    state = obj.optimizer_state_dict()
    other = optimizer(
        AttributedTriFairOptimizer,
        smart_start_count=0,
        uncertainty_resamples=8,
        max_idle_rounds=4,
    )
    with pytest.raises(ValueError, match="configuration mismatch"):
        other.load_optimizer_state_dict(state)
