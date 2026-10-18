"""Behavioral tests for cached stability, component edits and matched pilots."""

import numpy as np
import pandas as pd
import pytest
from promptolution.utils.prompt import Prompt
from tests.fakes import optimizer, FakeLLM
from tri_fair.optimizers.stable import StableTriFairOptimizer
from tri_fair.search.stability import search_stability, balanced_gold_demonstrations
from tri_fair.search.uncertainty import resample_indices
from tri_fair.cli.pilot_stable import main


def initialized():
    obj = optimizer(
        StableTriFairOptimizer, smart_start_count=0, uncertainty_resamples=8
    )
    obj._pre_optimization_loop()
    return obj


def test_stability_uses_only_cached_predictions_and_leaves_scores_unchanged():
    obj = initialized()
    p = obj.incumbents[0]
    blocks = obj.task.get_evaluated_blocks([p])[p]
    before = obj.predictor.llm.get_token_count().copy()
    record = obj.get_fairness_record(p, blocks)
    result = search_stability(obj, p, blocks)
    assert obj.predictor.llm.get_token_count() == before
    assert obj.get_fairness_record(p, blocks) == record
    assert result["quality"] == pytest.approx(record["quality"])
    assert result["unfairness"] == pytest.approx(record["fairness"])
    assert result["conservative_quality"] <= result["quality"]
    assert result["conservative_unfairness"] >= result["unfairness"]
    assert search_stability(obj, p, blocks) == result
    with pytest.raises(ValueError, match="new predictions"):
        search_stability(obj, p, range(obj.task.n_blocks))


def test_search_fold_sensitivity_needs_no_extra_inference():
    obj = initialized()
    p = obj.incumbents[0]
    obj._get_block_vectors([p], list(range(obj.task.n_blocks)))
    before = obj.predictor.llm.get_token_count().copy()
    result = search_stability(obj, p, range(obj.task.n_blocks))
    assert result["usable_folds"] == obj.task.n_blocks
    assert obj.predictor.llm.get_token_count() == before


def test_instruction_mutation_preserves_parent_demonstrations():
    obj = initialized()
    p = obj.incumbents[0]
    parent = Prompt(
        p.instruction,
        ["Input:\nGold example\nOutput:\n<final_answer>nurse</final_answer>"],
    )
    obj._get_block_vectors([parent], [0, 1, 2])
    obj.incumbents = [parent]
    children = obj._generate_challengers()
    assert any(
        c.instruction != parent.instruction and c.few_shots == parent.few_shots
        for c in children
    )
    assert any(
        c.instruction == parent.instruction and c.few_shots != parent.few_shots
        for c in children
    )


def test_cost_deletes_one_demo_without_fabricating_measured_scores():
    obj = initialized()
    p = Prompt(obj.incumbents[0].instruction, ["one", "two"])
    obj._get_block_vectors([p], [0, 1, 2])
    obj.incumbents = [p]
    obj.proposal_round = 4
    before = obj.predictor.llm.get_token_count().copy()
    children = obj._generate_challengers()
    assert any(
        c.instruction == p.instruction and c.few_shots == ["one"] for c in children
    )
    assert obj.predictor.llm.get_token_count() == before
    assert all(not obj.task.get_evaluated_blocks([c])[c] for c in children)


def test_fewshot_selection_preserves_official_winobias_pair():
    obj = initialized()
    obj.df_few_shots = pd.DataFrame(
        [
            dict(
                input="The doctor thanked the nurse because she helped.",
                target="nurse",
                pair_id="p1",
                stereotype="pro",
                gender=1,
            ),
            dict(
                input="The doctor thanked the nurse because he helped.",
                target="nurse",
                pair_id="p1",
                stereotype="anti",
                gender=0,
            ),
        ]
    )
    before = obj.predictor.llm.get_token_count()
    demos = balanced_gold_demonstrations(obj, {"evidence": [], "confusions": []})
    assert len(demos) == 2
    assert any("she helped" in d for d in demos) and any(
        "he helped" in d for d in demos
    )
    assert obj.predictor.llm.get_token_count() == before


def test_winobias_bootstrap_keeps_whole_related_families():
    rows = pd.DataFrame(
        [dict(_review_id=f"{f}-{j}", template_id=f) for f in range(5) for j in range(8)]
    )
    a = resample_indices(rows, metric="winobias", seed=72, repeats=16)
    b = resample_indices(rows, metric="winobias", seed=72, repeats=16)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    for sample in a:
        counts = pd.Series(sample).value_counts()
        for family in range(5):
            assert (
                len(
                    set(
                        int(counts.get(i, 0))
                        for i in range(family * 8, (family + 1) * 8)
                    )
                )
                == 1
            )


def test_stable_optimize_checkpoint_and_resource_limits():
    obj = optimizer(
        StableTriFairOptimizer, smart_start_count=0, uncertainty_resamples=8
    )
    obj.optimize(4)
    assert obj._completed_steps == 4 and obj.generation_calls > 0
    assert any(r.get("measurements") for r in obj.proposal_memory)
    state = obj.optimizer_state_dict()
    other = optimizer(
        StableTriFairOptimizer, smart_start_count=0, uncertainty_resamples=8
    )
    other.load_optimizer_state_dict(state)
    assert other.proposal_round == obj.proposal_round
    bad = optimizer(StableTriFairOptimizer, smart_start_count=0, stability_strength=0.5)
    with pytest.raises(ValueError, match="Stability checkpoint"):
        bad.load_optimizer_state_dict(state)


def test_request_guidance_and_budget_include_the_complete_prompt():
    class RecordingLLM(FakeLLM):
        def get_response(self, prompts, system_prompts=None):
            self.requests = prompts
            return super().get_response(prompts, system_prompts)

    obj = initialized()
    obj.meta_llm = RecordingLLM()
    obj.task.fairness_metric_name = "winobias"
    obj._generate_challengers()
    assert all("Preserve pronouns" in p for p in obj.meta_llm.requests)
    assert all(
        len(obj.meta_llm.tokenizer.encode(p)) <= obj.generation_input_token_limit
        for p in obj.meta_llm.requests
    )
    assert all("search_block_sensitivity" in p for p in obj.meta_llm.requests)


def test_four_dataset_pilot_uses_shared_settings_and_no_hybrid(capsys):
    for dataset in ("bbq", "civil_comments", "bias_in_bios", "winobias"):
        assert main(["--dataset", dataset]) == 0
        output = capsys.readouterr().out
        commands = [
            line for line in output.splitlines() if "tri_fair.cli.train" in line
        ]
        assert len(commands) == 3
        for command in commands:
            assert "--n-init-prompts 4" in command
            assert "--budget-per-run 2000000" in command
            assert "--final-search-candidates 4" in command
            assert "--max-few-shot-examples 2" in command
            assert "--structured-labels" in command
            assert "tri_fair_v85_pilot_2000000" in command
        assert "Hybrid" not in output and "Tri-Fair-v8.5" in output
