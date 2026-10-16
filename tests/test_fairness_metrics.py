import numpy as np
import pandas as pd

from tri_fair.fairness.bbq import compute_bbq_fairness
from tri_fair.fairness.bios import compute_bios_fairness
from tri_fair.fairness.civilcomments import compute_civilcomments_fairness


def test_bbq_directional_and_ambiguity_adjusted_score():
    metadata = pd.DataFrame(
        {
            "category": ["Age"] * 4,
            "context_condition": ["ambig", "ambig", "disambig", "disambig"],
            "question_polarity": ["neg", "nonneg", "neg", "nonneg"],
            "target_loc": [0, 1, 0, 1],
            "unknown_loc": [2, 2, 2, 2],
        }
    )
    y_true = ["c", "c", "a", "b"]
    y_pred = ["a", "b", "a", "b"]
    result = compute_bbq_fairness(
        y_true,
        y_pred,
        metadata,
        min_group_count=1,
        require_all_contexts=True,
    )
    assert result.ready
    assert np.isclose(result.loss, 1.0)
    assert np.isclose(result.diagnostics["worst_abs_bias"], 1.0)


def test_bias_in_bios_rms_tpr_gap():
    metadata = pd.DataFrame({"gender": [0, 0, 1, 1, 0, 0, 1, 1]})
    y_true = ["nurse"] * 4 + ["teacher"] * 4
    y_pred = [
        "nurse",
        "nurse",
        "nurse",
        "nurse",
        "teacher",
        "teacher",
        "nurse",
        "nurse",
    ]
    result = compute_bios_fairness(
        y_true,
        y_pred,
        metadata,
        min_group_count=2,
        min_valid_professions=2,
    )
    assert result.ready
    assert np.isclose(result.loss, np.sqrt((0.0**2 + 1.0**2) / 2.0))


def test_civil_comments_worst_group_accuracy():
    metadata = pd.DataFrame(
        {
            "male": [1, 1, 0, 0],
            "female": [0, 0, 1, 1],
        }
    )
    y_true = ["non_toxic", "toxic", "non_toxic", "toxic"]
    y_pred = ["non_toxic", "toxic", "toxic", "toxic"]
    result = compute_civilcomments_fairness(
        y_true,
        y_pred,
        metadata,
        min_group_count=1,
        min_valid_groups=4,
        identity_columns=("male", "female"),
    )
    assert result.ready
    assert np.isclose(result.diagnostics["worst_group_accuracy"], 0.0)
    assert np.isclose(result.loss, 1.0)


def test_bbq_official_target_metadata_overrides_fallback(tmp_path):
    from tri_fair.fairness.bbq import attach_official_target_locations

    metadata_path = tmp_path / "additional_metadata.csv"
    pd.DataFrame(
        {
            "category": ["Age"],
            "question_index": ["official-question-index"],
            "example_id": [0],
            "target_loc": [1],
        }
    ).to_csv(metadata_path, index=False)
    frame = pd.DataFrame(
        {
            "category": ["Age"],
            "question_index": ["converted-question-index"],
            "example_id": [0],
            "target_loc": [0],
        }
    )
    result = attach_official_target_locations(
        frame,
        cache_path=metadata_path,
        require_official=True,
    )
    assert int(result.loc[0, "target_loc"]) == 1
    assert result.loc[0, "target_loc_source"] == "official"


def test_bbq_strict_mode_drops_complete_unscored_template_group(tmp_path):
    from tri_fair.fairness.bbq import attach_official_target_locations

    metadata_path = tmp_path / "additional_metadata.csv"
    pd.DataFrame(
        {
            "category": ["Age", "Age", "Age"],
            "example_id": [0, 1, 4],
            "target_loc": [1, np.nan, 2],
        }
    ).to_csv(metadata_path, index=False)

    frame = pd.DataFrame(
        {
            "category": ["Age", "Age", "Age"],
            "question_index": ["q0", "q0", "q1"],
            "example_id": [0, 1, 4],
            "template_id": ["Age:0", "Age:0", "Age:1"],
            "target_loc": [0, 0, 0],
        }
    )

    result = attach_official_target_locations(
        frame,
        cache_path=metadata_path,
        require_official=True,
    )

    assert result["example_id"].tolist() == [4]
    assert result["template_id"].tolist() == ["Age:1"]
    assert result["target_loc"].tolist() == [2]
    assert result["target_loc_source"].tolist() == ["official"]
