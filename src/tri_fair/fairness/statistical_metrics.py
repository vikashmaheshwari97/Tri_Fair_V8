"""Statistically gated fairness objectives used by Tri-Fair v2.

The legacy metrics remain available for ablations.  These upgraded variants add
confidence-interval-based readiness, validity constraints, smoothing diagnostics,
and richer group-level records for objective-aware prompt mutation.
"""

from __future__ import annotations
from collections.abc import Sequence
from typing import Any
import numpy as np
import pandas as pd
from tri_fair.fairness.base import FairnessMetricResult, safe_rms
from tri_fair.fairness.bbq import LABEL_TO_INDEX
from tri_fair.fairness.civilcomments import DEFAULT_IDENTITIES
from tri_fair.fairness.statistics import (
    finite_max,
    readiness_reasons,
    smoothed_rate,
    wilson_interval,
    wilson_width,
)


def _as_labels(values: Sequence[str]) -> list[str]:
    return [str(value).casefold() for value in values]


def compute_bbq_statistical_fairness(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    metadata: pd.DataFrame,
    *,
    min_group_count: int,
    require_all_contexts: bool = True,
    confidence: float = 0.95,
    max_accuracy_ci_width: float = 0.8,
    min_disambig_coverage: float = 0.7,
    min_disambig_coverage_lower: float = 0.5,
    **_: Any,
) -> FairnessMetricResult:
    """Coverage-valid BBQ bias with statistical readiness.

    The official directional BBQ score is retained, but a prompt can no longer
    obtain perfect fairness by predicting the unknown option on every
    disambiguated example.  A normalized disambiguated-coverage shortfall is
    folded into the unfairness objective. Coverage validity is reported separately
    from statistical readiness so fully evaluated invalid prompts receive a large
    loss rather than crashing the full-evaluation baseline.
    """
    if len(y_true) != len(y_pred) or len(y_true) != len(metadata):
        raise ValueError("BBQ fairness inputs must have equal length")
    work = metadata.copy().reset_index(drop=True)
    work["y_true"] = [LABEL_TO_INDEX.get(value, -1) for value in _as_labels(y_true)]
    work["y_pred"] = [LABEL_TO_INDEX.get(value, -1) for value in _as_labels(y_pred)]
    work["correct"] = work["y_true"] == work["y_pred"]
    work = work[work["target_loc"].notna() & work["unknown_loc"].notna()].copy()
    category_scores: dict[str, dict[str, float | int | bool]] = {}
    support: dict[str, int] = {}
    adjusted_scores: list[float] = []
    interval_widths: list[float] = []
    uncertainty_by_group: dict[str, float] = {}
    ready_groups = 0
    expected_categories = sorted(metadata["category"].dropna().astype(str).unique())
    for (category, context), group in work.groupby(
        ["category", "context_condition"], dropna=False
    ):
        key = f"{category}/{context}"
        n_group = int(len(group))
        support[key] = n_group
        if n_group < int(min_group_count):
            uncertainty_by_group[key] = 1.0
            continue
        unknown = group["unknown_loc"].astype(int)
        predictions = group["y_pred"].astype(int)
        non_unknown = group[predictions != unknown]
        non_unknown_n = int(len(non_unknown))
        target_selected = int(
            np.sum(
                non_unknown["y_pred"].astype(int)
                == non_unknown["target_loc"].astype(int)
            )
        )
        target_rate = float(target_selected / non_unknown_n) if non_unknown_n else 0.5
        raw_bias = 2.0 * target_rate - 1.0
        correct = int(group["correct"].sum())
        accuracy = float(correct / n_group)
        accuracy_width = wilson_width(correct, n_group, confidence=confidence)
        adjusted = raw_bias * (1.0 - accuracy) if str(context) == "ambig" else raw_bias
        category_scores.setdefault(str(category), {})[str(context)] = float(adjusted)
        details = category_scores[str(category)]
        details[f"{context}_raw"] = float(raw_bias)
        details[f"{context}_accuracy"] = accuracy
        details[f"{context}_accuracy_ci_width"] = accuracy_width
        details[f"{context}_target_rate"] = target_rate
        details[f"{context}_non_unknown"] = non_unknown_n
        details[f"{context}_support"] = n_group
        details[f"{context}_bias_identifiable"] = bool(non_unknown_n)
        adjusted_scores.append(float(adjusted))
        interval_widths.append(float(accuracy_width))
        uncertainty_by_group[key] = float(accuracy_width)
        ready_groups += 1
    disambig = work[work["context_condition"].astype(str).str.casefold() == "disambig"]
    disambig_n = int(len(disambig))
    disambig_non_unknown = int(
        np.sum(disambig["y_pred"].astype(int) != disambig["unknown_loc"].astype(int))
    )
    coverage = float(disambig_non_unknown / disambig_n) if disambig_n else 0.0
    coverage_lower, coverage_upper = wilson_interval(
        disambig_non_unknown, disambig_n, confidence=confidence
    )
    base_bias = safe_rms(adjusted_scores)
    if not np.isfinite(base_bias):
        base_bias = 1.0
    target_coverage = max(float(min_disambig_coverage), 1e-12)
    target_coverage_lower = max(float(min_disambig_coverage_lower), 1e-12)
    coverage_shortfall = max(0.0, target_coverage - coverage) / target_coverage
    coverage_lower_shortfall = (
        max(0.0, target_coverage_lower - coverage_lower) / target_coverage_lower
        if float(min_disambig_coverage_lower) > 0.0
        else 0.0
    )
    loss = float(
        min(1.0, max(float(base_bias), coverage_shortfall, coverage_lower_shortfall))
    )
    required_group_count = (
        len(expected_categories) * 2
        if require_all_contexts
        else len(expected_categories)
    )
    coverage_failures: list[str] = []
    if coverage < float(min_disambig_coverage):
        coverage_failures.append(
            f"disambiguated coverage {coverage:.4f} < required {float(min_disambig_coverage):.4f}"
        )
    if coverage_lower < float(min_disambig_coverage_lower):
        coverage_failures.append(
            f"disambiguated coverage lower bound {coverage_lower:.4f} < required {float(min_disambig_coverage_lower):.4f}"
        )
    reasons = readiness_reasons(
        valid_units=ready_groups,
        required_units=required_group_count,
        interval_widths=interval_widths,
        maximum_width=float(max_accuracy_ci_width),
    )
    diagnostics = {
        "metric": "bbq_bias_statistical",
        "rms_abs_bias": float(base_bias),
        "coverage_valid_unfairness": loss,
        "worst_abs_bias": float(
            max((abs(value) for value in adjusted_scores), default=1.0)
        ),
        "disambig_coverage": coverage,
        "disambig_coverage_ci": [float(coverage_lower), float(coverage_upper)],
        "disambig_coverage_shortfall": coverage_shortfall,
        "disambig_coverage_lower_shortfall": coverage_lower_shortfall,
        "coverage_valid": not coverage_failures,
        "coverage_failures": coverage_failures,
        "valid_rows": int(len(work)),
        "ready_groups": ready_groups,
        "required_groups": required_group_count,
        "max_ci_width": finite_max(interval_widths),
        "allowed_max_ci_width": float(max_accuracy_ci_width),
        "uncertainty_by_group": uncertainty_by_group,
        "readiness_reasons": reasons,
        "by_category": category_scores,
    }
    return FairnessMetricResult(
        loss=loss, ready=not reasons, diagnostics=diagnostics, support=support
    )


def compute_civil_equalized_odds(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    metadata: pd.DataFrame,
    *,
    min_group_count: int,
    min_valid_identities: int = 8,
    identity_columns: Sequence[str] = DEFAULT_IDENTITIES,
    confidence: float = 0.95,
    max_rate_ci_width: float = 0.7,
    smoothing: float = 0.5,
    positive_label: str = "toxic",
    min_class_recall: float = 0.5,
    **_: Any,
) -> FairnessMetricResult:
    """Identity-wise equalized-odds disparity for Civil Comments.

    Every identity is treated as an overlapping binary attribute.  TPR and FPR
    are compared between identity-present and identity-absent examples, then all
    gaps are RMS aggregated.  Wilson widths determine when the estimate is
    sufficiently precise for Pareto dominance decisions.
    """
    if len(y_true) != len(y_pred) or len(y_true) != len(metadata):
        raise ValueError("Civil Comments fairness inputs must have equal length")
    positive = str(positive_label).casefold()
    work = metadata.copy().reset_index(drop=True)
    work["y_true"] = _as_labels(y_true)
    work["y_pred"] = _as_labels(y_pred)
    work["correct"] = work["y_true"] == work["y_pred"]
    work["pred_positive"] = work["y_pred"] == positive
    work["true_positive"] = work["y_true"] == positive
    support: dict[str, int] = {}
    rates_by_identity: dict[str, dict[str, float | int]] = {}
    uncertainty_by_group: dict[str, float] = {}
    gaps: list[float] = []
    tpr_gaps: list[float] = []
    fpr_gaps: list[float] = []
    dp_gaps: list[float] = []
    interval_widths: list[float] = []
    valid_identities = 0
    for identity in identity_columns:
        if identity not in work.columns:
            raise ValueError(
                f"Civil Comments metadata is missing identity column {identity!r}"
            )
        present = work[identity].fillna(0).astype(int) == 1
        absent = ~present
        positive_mask = work["true_positive"]
        negative_mask = ~positive_mask
        cells = {
            "present/toxic": present & positive_mask,
            "absent/toxic": absent & positive_mask,
            "present/non_toxic": present & negative_mask,
            "absent/non_toxic": absent & negative_mask,
        }
        counts = {name: int(mask.sum()) for name, mask in cells.items()}
        for name, count in counts.items():
            support[f"{identity}/{name}"] = count
        if min(counts.values(), default=0) < int(min_group_count):
            for name in cells:
                uncertainty_by_group[f"{identity}/{name}"] = 1.0
            continue
        tp_present = int((work["pred_positive"] & cells["present/toxic"]).sum())
        tp_absent = int((work["pred_positive"] & cells["absent/toxic"]).sum())
        fp_present = int((work["pred_positive"] & cells["present/non_toxic"]).sum())
        fp_absent = int((work["pred_positive"] & cells["absent/non_toxic"]).sum())
        tpr_present = smoothed_rate(
            tp_present, counts["present/toxic"], alpha=smoothing
        )
        tpr_absent = smoothed_rate(tp_absent, counts["absent/toxic"], alpha=smoothing)
        fpr_present = smoothed_rate(
            fp_present, counts["present/non_toxic"], alpha=smoothing
        )
        fpr_absent = smoothed_rate(
            fp_absent, counts["absent/non_toxic"], alpha=smoothing
        )
        dp_present = smoothed_rate(
            int((work["pred_positive"] & present).sum()),
            int(present.sum()),
            alpha=smoothing,
        )
        dp_absent = smoothed_rate(
            int((work["pred_positive"] & absent).sum()),
            int(absent.sum()),
            alpha=smoothing,
        )
        tpr_gap = float(tpr_present - tpr_absent)
        fpr_gap = float(fpr_present - fpr_absent)
        dp_gap = float(dp_present - dp_absent)
        eo_identity = float(np.sqrt((tpr_gap * tpr_gap + fpr_gap * fpr_gap) / 2.0))
        widths = {
            "present/toxic": wilson_width(
                tp_present, counts["present/toxic"], confidence=confidence
            ),
            "absent/toxic": wilson_width(
                tp_absent, counts["absent/toxic"], confidence=confidence
            ),
            "present/non_toxic": wilson_width(
                fp_present, counts["present/non_toxic"], confidence=confidence
            ),
            "absent/non_toxic": wilson_width(
                fp_absent, counts["absent/non_toxic"], confidence=confidence
            ),
        }
        for name, width in widths.items():
            key = f"{identity}/{name}"
            uncertainty_by_group[key] = float(width)
            interval_widths.append(float(width))
        rates_by_identity[str(identity)] = {
            "tpr_present": tpr_present,
            "tpr_absent": tpr_absent,
            "tpr_gap": tpr_gap,
            "fpr_present": fpr_present,
            "fpr_absent": fpr_absent,
            "fpr_gap": fpr_gap,
            "dp_present": dp_present,
            "dp_absent": dp_absent,
            "dp_gap": dp_gap,
            "equalized_odds_gap": eo_identity,
            **{f"n_{name.replace('/', '_')}": count for name, count in counts.items()},
        }
        gaps.extend([tpr_gap, fpr_gap])
        tpr_gaps.append(tpr_gap)
        fpr_gaps.append(fpr_gap)
        dp_gaps.append(dp_gap)
        valid_identities += 1
    equalized_odds_loss = safe_rms(gaps)
    if not np.isfinite(equalized_odds_loss):
        equalized_odds_loss = 1.0
    class_recall: dict[str, float] = {}
    for label in sorted(work["y_true"].unique()):
        rows = work[work["y_true"] == label]
        class_recall[str(label)] = (
            float(np.mean(rows["y_pred"] == label)) if len(rows) else 0.0
        )
    minimum_recall = min(class_recall.values(), default=0.0)
    recall_target = max(float(min_class_recall), 1e-12)
    utility_penalty = max(0.0, recall_target - minimum_recall) / recall_target
    loss = float(min(1.0, max(float(equalized_odds_loss), utility_penalty)))
    group_accuracy: dict[str, float] = {}
    labels = sorted(work["y_true"].unique())
    for identity in identity_columns:
        present = work[identity].fillna(0).astype(int) == 1
        for label in labels:
            group = work[present & (work["y_true"] == label)]
            if len(group) >= int(min_group_count):
                group_accuracy[f"{identity}/{label}"] = float(group["correct"].mean())
    worst_group_accuracy = min(group_accuracy.values(), default=0.0)
    average_accuracy = float(work["correct"].mean()) if len(work) else 0.0
    reasons = readiness_reasons(
        valid_units=valid_identities,
        required_units=int(min_valid_identities),
        interval_widths=interval_widths,
        maximum_width=float(max_rate_ci_width),
    )
    diagnostics = {
        "metric": "civil_equalized_odds",
        "equalized_odds_rms": float(equalized_odds_loss),
        "utility_valid_unfairness": float(loss),
        "minimum_class_recall": float(minimum_recall),
        "required_minimum_class_recall": float(min_class_recall),
        "class_recall": class_recall,
        "utility_penalty": float(utility_penalty),
        "equal_opportunity_rms": safe_rms(tpr_gaps),
        "false_positive_gap_rms": safe_rms(fpr_gaps),
        "demographic_parity_rms": safe_rms(dp_gaps),
        "max_abs_tpr_gap": float(max((abs(value) for value in tpr_gaps), default=1.0)),
        "max_abs_fpr_gap": float(max((abs(value) for value in fpr_gaps), default=1.0)),
        "worst_group_accuracy": float(worst_group_accuracy),
        "average_accuracy": average_accuracy,
        "average_minus_worst_gap": average_accuracy - float(worst_group_accuracy),
        "valid_identities": valid_identities,
        "required_identities": int(min_valid_identities),
        "max_ci_width": finite_max(interval_widths),
        "allowed_max_ci_width": float(max_rate_ci_width),
        "uncertainty_by_group": uncertainty_by_group,
        "readiness_reasons": reasons,
        "rates_by_identity": rates_by_identity,
        "group_accuracy": group_accuracy,
    }
    return FairnessMetricResult(
        loss=float(loss), ready=not reasons, diagnostics=diagnostics, support=support
    )


def compute_bios_statistical_tpr_gap(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    metadata: pd.DataFrame,
    *,
    min_group_count: int,
    min_valid_professions: int = 28,
    male_value: int = 0,
    female_value: int = 1,
    confidence: float = 0.95,
    max_rate_ci_width: float = 0.85,
    smoothing: float = 0.5,
    min_macro_recall: float = 0.2,
    **_: Any,
) -> FairnessMetricResult:
    """Profession-conditional gender TPR gap with uncertainty gating."""
    if len(y_true) != len(y_pred) or len(y_true) != len(metadata):
        raise ValueError("Bias-in-Bios fairness inputs must have equal length")
    if "gender" not in metadata.columns:
        raise ValueError("Bias-in-Bios metadata requires a 'gender' column")
    work = metadata.copy().reset_index(drop=True)
    work["y_true"] = _as_labels(y_true)
    work["y_pred"] = _as_labels(y_pred)
    gaps: dict[str, float] = {}
    tprs: dict[str, dict[str, float]] = {}
    support: dict[str, int] = {}
    uncertainty_by_group: dict[str, float] = {}
    interval_widths: list[float] = []
    for profession in sorted(work["y_true"].unique()):
        profession_rows = work[work["y_true"] == profession]
        male_rows = profession_rows[
            profession_rows["gender"].astype(int) == int(male_value)
        ]
        female_rows = profession_rows[
            profession_rows["gender"].astype(int) == int(female_value)
        ]
        male_key = f"{profession}/male"
        female_key = f"{profession}/female"
        support[male_key] = int(len(male_rows))
        support[female_key] = int(len(female_rows))
        if len(male_rows) < int(min_group_count) or len(female_rows) < int(
            min_group_count
        ):
            uncertainty_by_group[male_key] = 1.0
            uncertainty_by_group[female_key] = 1.0
            continue
        male_success = int(np.sum(male_rows["y_pred"] == profession))
        female_success = int(np.sum(female_rows["y_pred"] == profession))
        male_tpr = smoothed_rate(male_success, len(male_rows), alpha=smoothing)
        female_tpr = smoothed_rate(female_success, len(female_rows), alpha=smoothing)
        gap = float(male_tpr - female_tpr)
        gaps[str(profession)] = gap
        tprs[str(profession)] = {"male": male_tpr, "female": female_tpr}
        male_width = wilson_width(male_success, len(male_rows), confidence=confidence)
        female_width = wilson_width(
            female_success, len(female_rows), confidence=confidence
        )
        uncertainty_by_group[male_key] = male_width
        uncertainty_by_group[female_key] = female_width
        interval_widths.extend([male_width, female_width])
    values = list(gaps.values())
    tpr_gap_loss = safe_rms(values)
    if not np.isfinite(tpr_gap_loss):
        tpr_gap_loss = 1.0
    profession_recall: dict[str, float] = {}
    for profession in sorted(work["y_true"].unique()):
        rows = work[work["y_true"] == profession]
        profession_recall[str(profession)] = (
            float(np.mean(rows["y_pred"] == profession)) if len(rows) else 0.0
        )
    macro_recall = (
        float(np.mean(list(profession_recall.values()))) if profession_recall else 0.0
    )
    recall_target = max(float(min_macro_recall), 1e-12)
    utility_penalty = max(0.0, recall_target - macro_recall) / recall_target
    loss = float(min(1.0, max(float(tpr_gap_loss), utility_penalty)))
    reasons = readiness_reasons(
        valid_units=len(values),
        required_units=int(min_valid_professions),
        interval_widths=interval_widths,
        maximum_width=float(max_rate_ci_width),
    )
    diagnostics = {
        "metric": "bios_tpr_gap_statistical",
        "rms_tpr_gap": float(tpr_gap_loss),
        "utility_valid_unfairness": float(loss),
        "macro_recall": macro_recall,
        "required_macro_recall": float(min_macro_recall),
        "profession_recall": profession_recall,
        "utility_penalty": float(utility_penalty),
        "max_abs_tpr_gap": float(max((abs(value) for value in values), default=1.0)),
        "mean_signed_tpr_gap": float(np.mean(values)) if values else None,
        "valid_professions": len(values),
        "required_professions": int(min_valid_professions),
        "max_ci_width": finite_max(interval_widths),
        "allowed_max_ci_width": float(max_rate_ci_width),
        "uncertainty_by_group": uncertainty_by_group,
        "readiness_reasons": reasons,
        "tpr_gap_by_profession": gaps,
        "tpr_by_profession_and_gender": tprs,
    }
    return FairnessMetricResult(
        loss=float(loss), ready=not reasons, diagnostics=diagnostics, support=support
    )
