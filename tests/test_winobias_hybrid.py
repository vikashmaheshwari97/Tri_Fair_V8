"""Dataset leakage, hybrid search and honest indicator integration checks."""

import copy
import random
import numpy as np
import pandas as pd
import pytest
from tests.fakes import optimizer
from tri_fair.config.datasets import ALL_DATASETS
from tri_fair.data.loading import (
    create_dev_tasks,
    create_test_task,
    load_or_create_manifest,
)
from tri_fair.data.partitions import split_search_validation
from tri_fair.data.winobias import load_winobias_sources, validate_partitions
from tri_fair.fairness.winobias import compute_winobias_fairness
from tri_fair.optimizers.hybrid import HybridTriFairOptimizer
from tri_fair.optimizers.nsga2 import NSGA2Fair
from tri_fair.reporting.pilot_indicators import shortlist_indicators
from tri_fair.reporting.front_report import analyse


@pytest.mark.parametrize("seed", [72, 73, 74])
def test_official_data_prepare_reuse_and_protected_partitions(tmp_path, seed):
    config = ALL_DATASETS["winobias"]
    dev, pool = create_dev_tasks(
        config, "sequential_block", seed=seed, manifest_dir=tmp_path
    )
    test = create_test_task(config, "full", seed=seed, manifest_dir=tmp_path)
    search, validation = split_search_validation(dev)
    groups = [set(t.df.template_id) for t in [search, validation, test]] + [
        set(pool.template_id)
    ]
    for i, left in enumerate(groups):
        for right in groups[i + 1 :]:
            assert not left & right
    for frame in [search.df, validation.df, test.df, pool]:
        assert frame.groupby("pair_id").size().eq(2).all()
        assert not frame.input.str.contains(r"\[|\]").any()
        result = compute_winobias_fairness(
            frame.target, frame.target, frame, min_group_count=8
        )
        assert result.ready and result.loss == 0
    sources = load_winobias_sources()
    assert len(sources["dev"]) == 1576 and len(sources["test"]) == 1584
    manifest, path = load_or_create_manifest(
        config, sources, seed=seed, manifest_dir=tmp_path
    )
    tampered = copy.deepcopy(manifest)
    tampered.blocks[0][0], tampered.blocks[1][0] = (
        tampered.blocks[1][0],
        tampered.blocks[0][0],
    )
    with pytest.raises(ValueError, match="family split"):
        validate_partitions(sources, tampered.partitions, tampered.blocks)


def test_fairness_captures_stereotype_gap_and_chance_collapse():
    frame = load_winobias_sources()["dev"]
    wrong = frame.target.map({"a": "b", "b": "a"})
    predictions = np.where(frame.stereotype == "pro", frame.target, wrong)
    biased = compute_winobias_fairness(
        frame.target, predictions, frame, min_group_count=8
    )
    assert biased.ready and biased.loss == 1
    failed = compute_winobias_fairness(frame.target, wrong, frame, min_group_count=8)
    assert failed.ready and failed.loss == 1 and failed.diagnostics["raw_gap_rms"] == 0
    constant = compute_winobias_fairness(
        frame.target, ["a"] * len(frame), frame, min_group_count=8
    )
    assert constant.loss == 1 and constant.diagnostics["recall_by_label"]["b"] == 0
    assert not compute_winobias_fairness(
        frame.target.iloc[:1], frame.target.iloc[:1], frame.iloc[:1], min_group_count=8
    ).ready


def test_hybrid_reuses_full_evaluation_and_standard_environmental_selection():
    assert (
        HybridTriFairOptimizer._environmental_selection
        is NSGA2Fair._environmental_selection
    )
    random.seed(72)
    np.random.seed(72)
    obj = optimizer(HybridTriFairOptimizer)
    obj.optimize(6)
    assert obj.portfolio_usage["evolution"] >= 1
    assert obj.portfolio_usage["reflection"] >= 1
    assert obj.portfolio_usage["compression"] >= 1
    for prompt in obj.incumbents:
        assert len(obj.task.get_evaluated_blocks([prompt])[prompt]) == obj.task.n_blocks
    assert obj.search_diagnostics()["full_search_evaluation"]
    assert obj.generation_calls > 0


def test_hybrid_checkpoint_tracks_portfolio_and_rejects_mismatched_config():
    obj = optimizer(HybridTriFairOptimizer)
    obj.optimize(3)
    state = obj.optimizer_state_dict()
    other = optimizer(HybridTriFairOptimizer)
    other.load_optimizer_state_dict(state)
    assert other.portfolio_round == obj.portfolio_round
    assert other.proposal_memory == obj.proposal_memory
    mismatch = optimizer(HybridTriFairOptimizer, prompt_word_limit=80)
    with pytest.raises(ValueError, match="configuration"):
        mismatch.load_optimizer_state_dict(state)


def test_search_indicators_do_not_claim_heldout_nr2():
    run = dict(
        status="measured",
        comparable=True,
        final_incumbent_candidates=[dict(quality=0.8, cost=20, fairness=0.1)],
        dataset_configuration={
            "fairness": {"fairness_kwargs": {"normalization_cost_upper_bound": 100}}
        },
    )
    value = shortlist_indicators(run)
    assert value["search_hv_shortlist_3d"] == pytest.approx(0.9 * 0.9 * 1.0)
    assert value["noisy_r2_3d"] is None
    assert value["search_r2_shortlist_3d"] > 0


def test_heldout_report_requires_alignment_and_scores_true_nr2():
    frame = pd.DataFrame(
        [
            dict(
                dataset="winobias",
                model="qwen-3-30b",
                optimizer="Tri-Fair-v8.4-Hybrid",
                seed=72,
                prompt_id="p",
                prompt="prompt",
                chosen_step=2,
                actual_budget_tokens=1000,
                evaluation_split="validation",
                dev_quality=0.9,
                dev_cost=20,
                dev_fairness=0.1,
                dev_fairness_ready=True,
                validation_quality=0.8,
                validation_cost=20,
                validation_fairness=0.2,
                validation_fairness_ready=True,
            )
        ]
    )
    result = analyse(frame, split="validation", budget=1000000)
    assert result.noisy_r2_3d.iloc[0] > 0
    with pytest.raises(ValueError, match="split"):
        analyse(frame, split="test", budget=1000000)
    with pytest.raises(ValueError, match="columns"):
        analyse(
            frame.drop(columns="validation_quality"), split="validation", budget=1000000
        )


def test_failed_pilot_export_remains_unavailable(tmp_path):
    from tri_fair.reporting.pilot_indicators import export_report

    result = export_report(
        dict(
            dataset="winobias",
            model="qwen-3-30b",
            seed=72,
            runs=[
                dict(method="Tri-Fair-v8.4-Hybrid", status="failed", comparable=False)
            ],
        ),
        tmp_path,
    )
    assert result["rows"][0]["status"] == "unavailable"
    assert result["rows"][0]["accuracy"] is None
    assert (tmp_path / "pilot_comparison.tex").is_file()
