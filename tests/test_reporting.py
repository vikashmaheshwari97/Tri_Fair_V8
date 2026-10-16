import pandas as pd
import pytest
from tri_fair.reporting.summarize import summarize


def validation_frame():
    return pd.DataFrame(
        [
            dict(
                dataset="bbq",
                model="qwen-3-30b",
                optimizer="Tri-Fair-v8",
                seed=72,
                chosen_step=2,
                actual_budget_tokens=900,
                prompt_id=p,
                evaluation_split="validation",
                validation_quality=q,
                validation_cost=c,
                validation_fairness=f,
                validation_fairness_ready=True,
            )
            for p, q, c, f in [("a", 0.9, 10, 0.1), ("b", 0.5, 80, 0.8)]
        ]
    )


def test_representative_is_frozen_on_validation_not_test():
    frame = validation_frame()
    _, _, selection = summarize(frame, budget=1000)
    assert selection[0]["prompt_id"] == "a"
    test = frame.rename(
        columns={
            c: c.replace("validation_", "test_")
            for c in frame
            if c.startswith("validation_")
        }
    )
    test["evaluation_split"] = "test"
    test.loc[test.prompt_id == "a", "test_quality"] = 0.1
    test.loc[test.prompt_id == "b", "test_quality"] = 1.0
    result, _, _ = summarize(test, split="test", budget=1000, selections=selection)
    assert result.iloc[0].prompt_id == "a" and result.iloc[0].selected_quality == 0.1
    with pytest.raises(ValueError, match="frozen"):
        summarize(test, split="test")
    test["chosen_step"] = 3
    with pytest.raises(ValueError, match="checkpoint"):
        summarize(test, split="test", budget=1000, selections=selection)


def test_no_ready_validation_candidates_remain_visible():
    frame = validation_frame()
    frame["validation_fairness_ready"] = False
    per_seed, summary, selection = summarize(frame, budget=1000)
    assert not selection and per_seed.iloc[0].n_fairness_ready == 0
    assert per_seed.iloc[0].representative_status == "no_ready_candidate"
    assert summary.iloc[0].archive_hypervolume_mean == 0.0
