"""Pilot discovery must not silently select a different experiment."""

import json
import pandas as pd
import pytest
from tri_fair.cli.evaluate import _logging_dir, _membership_for_run
from tri_fair.cli.common import prompt_id
from promptolution.utils.prompt import Prompt


def make_pilot(root, stamp="run1"):
    directory = root / "bias_in_bios" / "seed72" / stamp
    method = "Tri-Fair-v8.4-Hybrid"
    run = directory / method / "run"
    run.mkdir(parents=True)
    (directory / "pilot_report.json").write_text(
        json.dumps(
            dict(
                dataset="bias_in_bios",
                seed=72,
                runs=[dict(method=method, status="measured")],
            )
        )
    )
    return directory, run


def test_pilot_discovery_from_model_root_and_exact_timestamp(tmp_path):
    directory, run = make_pilot(tmp_path)
    assert _logging_dir(tmp_path, "bias_in_bios", "Tri-Fair-v8.4-Hybrid", 72) == run
    assert _logging_dir(directory, "bias_in_bios", "Tri-Fair-v8.4-Hybrid", 72) == run
    with pytest.raises(ValueError, match="dataset/seed"):
        _logging_dir(directory, "bias_in_bios", "Tri-Fair-v8.4-Hybrid", 73)


def test_multiple_pilots_require_explicit_choice(tmp_path):
    first, run = make_pilot(tmp_path, "run1")
    make_pilot(tmp_path, "run2")
    with pytest.raises(ValueError, match="Multiple pilot"):
        _logging_dir(tmp_path, "bias_in_bios", "Tri-Fair-v8.4-Hybrid", 72)
    assert _logging_dir(first, "bias_in_bios", "Tri-Fair-v8.4-Hybrid", 72) == run


def test_standard_pointer_is_preserved_and_failed_pilot_rejected(tmp_path):
    _, run = make_pilot(tmp_path)
    pointer = (
        tmp_path / "bias_in_bios" / "NSGAII-PO-Fair" / "seed72" / "logging_dir.txt"
    )
    pointer.parent.mkdir(parents=True)
    pointer.write_text(str(run))
    assert _logging_dir(tmp_path, "bias_in_bios", "NSGAII-PO-Fair", 72) == run
    with pytest.raises(ValueError, match="measured run"):
        _logging_dir(tmp_path, "bias_in_bios", "NotPresent", 72)


def test_final_only_membership_uses_one_real_step_and_exact_search_scores(tmp_path):
    rows = []
    for step in [1, 2]:
        prompt = Prompt(f"Instruction {step}", [])
        text = prompt.construct_prompt()
        rows.append(
            dict(
                step=step,
                prompt=text,
                prompt_id=prompt_id(text),
                instruction=prompt.instruction,
                few_shots_json="[]",
                is_incumbent=True,
                quality=0.6 + step / 10,
                cost=step,
                fairness=0.2,
                fairness_ready=True,
                total_tokens_downstream=100 * step,
            )
        )
    pd.DataFrame(rows).to_parquet(tmp_path / "step_results.parquet")
    (tmp_path / "args.json").write_text(json.dumps({"budget_per_run": 1000}))
    args = dict(
        optimizer="Tri-Fair-v8.4-Hybrid",
        dataset="bias_in_bios",
        model="qwen-3-30b",
        seed=72,
        minimum_tokens=0,
        maximum_tokens=1000,
    )
    all_steps = _membership_for_run(tmp_path, **args)
    final = _membership_for_run(tmp_path, final_only=True, **args)
    assert len(all_steps) == 2 and len(final) == 1
    assert final.chosen_step.iloc[0] == 2
    assert final.dev_quality.iloc[0] == pytest.approx(0.8)
    assert final.actual_budget_tokens.iloc[0] == 200
