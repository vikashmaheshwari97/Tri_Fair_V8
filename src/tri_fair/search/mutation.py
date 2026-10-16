"""Objective-aware prompt variation shared by Tri-Fair and its NSGA-II baseline.

Both optimizers receive the same crossover and diagnostic-guided mutation
operators.  This preserves a fair algorithmic comparison: the methods differ in
multi-fidelity intensification, not in access to stronger prompt-generation
operators.
"""

from __future__ import annotations
import json
import random
from typing import Any, Mapping
import numpy as np
import pandas as pd
from promptolution.utils.capo_utils import build_few_shot_examples
from promptolution.utils.formatting import extract_from_tag
from promptolution.utils.prompt import Prompt
from promptolution.utils.templates import CAPO_FEWSHOT_TEMPLATE

DEFAULT_MUTATION_WEIGHTS = {
    "fairness": 0.35,
    "quality": 0.25,
    "cost": 0.2,
    "balanced": 0.15,
    "explore": 0.05,
}
_OBJECTIVE_TEMPLATE = "You are revising an instruction used by a frozen language model in a\nthree-objective prompt-optimization experiment.\n\nTask:\n<task>\n\nAllowed output labels:\n<labels>\n\nCurrent offspring instruction:\n<instruction>\n\nMutation objective:\n<goal>\n\nMeasured parent diagnostics:\n<diagnostics>\n\nRequirements:\n- Rewrite the instruction itself; do not answer any dataset example.\n- Preserve the exact required output-label format.\n- Do not use protected attributes as shortcuts.\n- Do not obtain apparent fairness by always abstaining, always selecting an\n  unknown option, or collapsing to one label.\n- Keep the instruction operational and concise.\n- Return only the rewritten instruction between <prompt> and </prompt> tags.\n"
_GOALS = {
    "fairness": "Reduce the measured group unfairness while preserving task accuracy. Directly address the worst groups or categories named in the diagnostics.",
    "quality": "Improve predictive accuracy or macro-F1 while avoiding new group disparities and preserving the required label-only output.",
    "cost": "Compress the instruction and discourage unnecessary explanation so recurring input/output token cost falls without sacrificing correctness or fairness.",
    "balanced": "Find a balanced revision that improves accuracy, recurring inference cost, and group fairness together.",
    "explore": "Create a semantically faithful but diverse alternative that may reveal a new region of the three-objective Pareto front.",
}


def _normalised_weights(
    values: Mapping[str, float] | None,
) -> tuple[list[str], list[float]]:
    raw = (
        {str(key): float(value) for key, value in values.items()}
        if values is not None
        else dict(DEFAULT_MUTATION_WEIGHTS)
    )
    modes = list(DEFAULT_MUTATION_WEIGHTS)
    weights = [max(0.0, float(raw.get(mode, 0.0))) for mode in modes]
    total = sum(weights)
    if total <= 0:
        raise ValueError("objective mutation weights must contain a positive value")
    return (modes, [value / total for value in weights])


def _record(optimizer: Any, prompt: Prompt) -> dict[str, Any]:
    try:
        value = optimizer.get_fairness_record(prompt)
    except Exception:
        return {}
    return dict(value or {})


def _guide_for_mode(
    optimizer: Any, mother: Prompt, father: Prompt, mode: str
) -> tuple[Prompt, dict[str, Any]]:
    candidates = [
        (mother, _record(optimizer, mother)),
        (father, _record(optimizer, father)),
    ]

    def number(record: Mapping[str, Any], name: str, default: float) -> float:
        try:
            value = float(record.get(name, default))
        except (TypeError, ValueError):
            return default
        return value if np.isfinite(value) else default

    if mode == "fairness":
        return max(candidates, key=lambda item: number(item[1], "fairness", -1.0))
    if mode == "cost":
        return max(candidates, key=lambda item: number(item[1], "cost", -1.0))
    if mode == "quality":
        return min(candidates, key=lambda item: number(item[1], "quality", 2.0))
    if mode == "balanced":
        return max(
            candidates,
            key=lambda item: (
                number(item[1], "fairness", 1.0)
                + number(item[1], "cost", 100.0) / 100.0
                + (1.0 - number(item[1], "quality", 0.0))
            ),
        )
    return random.choice(candidates)


def _compact_diagnostics(record: Mapping[str, Any], *, max_chars: int = 2600) -> str:
    diagnostics = dict(record.get("diagnostics") or {})
    payload: dict[str, Any] = {
        "quality": record.get("quality"),
        "cost": record.get("cost"),
        "unfairness": record.get("fairness"),
        "metric": diagnostics.get("metric"),
    }
    metric = str(diagnostics.get("metric", ""))
    if metric.startswith("bbq_bias"):
        by_category = diagnostics.get("by_category") or {}
        failures: list[tuple[float, str, str, float]] = []
        for category, values in by_category.items():
            if not isinstance(values, Mapping):
                continue
            for context in ("ambig", "disambig"):
                value = values.get(context)
                try:
                    score = float(value)
                except (TypeError, ValueError):
                    continue
                failures.append((abs(score), str(category), context, score))
        failures.sort(reverse=True)
        payload.update(
            {
                "disambig_coverage": diagnostics.get("disambig_coverage"),
                "worst_category_contexts": [
                    {"category": category, "context": context, "bias": score}
                    for _, category, context, score in failures[:4]
                ],
            }
        )
    elif metric == "civil_equalized_odds":
        rates = diagnostics.get("rates_by_identity") or {}
        ranked = []
        for identity, values in rates.items():
            if not isinstance(values, Mapping):
                continue
            try:
                gap = float(values.get("equalized_odds_gap", 0.0))
            except (TypeError, ValueError):
                gap = 0.0
            ranked.append((gap, str(identity), values))
        ranked.sort(reverse=True)
        payload["worst_identities"] = [
            {
                "identity": identity,
                "tpr_gap": values.get("tpr_gap"),
                "fpr_gap": values.get("fpr_gap"),
                "equalized_odds_gap": gap,
            }
            for gap, identity, values in ranked[:4]
        ]
    elif metric.startswith("bios_tpr_gap"):
        gaps = diagnostics.get("tpr_gap_by_profession") or {}
        ranked = sorted(
            (
                (abs(float(value)), str(profession), float(value))
                for profession, value in gaps.items()
            ),
            reverse=True,
        )
        payload["worst_professions"] = [
            {"profession": profession, "male_minus_female_tpr": value}
            for _, profession, value in ranked[:5]
        ]
    else:
        for key in (
            "worst_group_accuracy",
            "average_minus_worst_gap",
            "max_abs_tpr_gap",
            "max_abs_fpr_gap",
        ):
            if key in diagnostics:
                payload[key] = diagnostics[key]
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    return encoded[:max_chars]


def _crossover_with_provenance(optimizer: Any):
    prompts: list[str] = []
    few_shots: list[list[str]] = []
    parents: list[tuple[Prompt, Prompt]] = []
    for _ in range(int(optimizer.crossovers_per_iter)):
        mother, father = optimizer._tournament_selection(optimizer.prompts)
        meta_prompt = (
            optimizer.crossover_template.replace("<mother>", mother.instruction)
            .replace("<father>", father.instruction)
            .strip()
        )
        prompts.append(meta_prompt)
        combined = list(mother.few_shots) + list(father.few_shots)
        target_count = (len(mother.few_shots) + len(father.few_shots)) // 2
        examples = random.sample(combined, target_count) if combined else []
        few_shots.append(examples)
        parents.append((mother, father))
    responses = optimizer.meta_llm.get_response(prompts)
    if isinstance(responses, str):
        responses = [responses]
    children = []
    for response, examples in zip(responses, few_shots):
        instruction = extract_from_tag(response, "<prompt>", "</prompt>").strip()
        if not instruction:
            instruction = "Classify the input using the required output-label format."
        children.append(Prompt(instruction, list(examples)))
    return (children, parents)


def _targeted_row(optimizer: Any, diagnostics: Mapping[str, Any]) -> pd.Series | None:
    frame = optimizer.df_few_shots
    if frame is None or len(frame) == 0:
        return None
    candidates = frame
    metric = str(diagnostics.get("metric", ""))
    if metric.startswith("bbq_bias"):
        by_category = diagnostics.get("by_category") or {}
        ranked: list[tuple[float, str, str]] = []
        for category, values in by_category.items():
            if not isinstance(values, Mapping):
                continue
            for context in ("ambig", "disambig"):
                try:
                    score = abs(float(values.get(context)))
                except (TypeError, ValueError):
                    continue
                ranked.append((score, str(category), context))
        if ranked:
            _, category, context = max(ranked)
            if "category" in candidates.columns:
                candidates = candidates[candidates["category"].astype(str) == category]
            if "context_condition" in candidates.columns:
                candidates = candidates[
                    candidates["context_condition"].astype(str) == context
                ]
    elif metric == "civil_equalized_odds":
        rates = diagnostics.get("rates_by_identity") or {}
        ranked = []
        for identity, values in rates.items():
            if not isinstance(values, Mapping):
                continue
            try:
                gap = float(values.get("equalized_odds_gap", 0.0))
            except (TypeError, ValueError):
                gap = 0.0
            ranked.append((gap, str(identity), values))
        if ranked:
            _, identity, values = max(ranked)
            if identity in candidates.columns:
                candidates = candidates[candidates[identity].fillna(0).astype(int) == 1]
            tpr_gap = abs(float(values.get("tpr_gap", 0.0)))
            fpr_gap = abs(float(values.get("fpr_gap", 0.0)))
            desired_label = "toxic" if tpr_gap >= fpr_gap else "non_toxic"
            target_column = optimizer.task.y_column
            if target_column in candidates.columns:
                candidates = candidates[
                    candidates[target_column].astype(str).str.casefold()
                    == desired_label
                ]
    elif metric.startswith("bios_tpr_gap"):
        gaps = diagnostics.get("tpr_gap_by_profession") or {}
        if gaps:
            profession, gap = max(gaps.items(), key=lambda item: abs(float(item[1])))
            target_column = optimizer.task.y_column
            if target_column in candidates.columns:
                candidates = candidates[
                    candidates[target_column].astype(str).str.casefold()
                    == str(profession).casefold()
                ]
            if "gender" in candidates.columns:
                desired_gender = 1 if float(gap) > 0 else 0
                candidates = candidates[
                    candidates["gender"].astype(int) == desired_gender
                ]
    if len(candidates) == 0:
        candidates = frame
    index = int(np.random.randint(0, len(candidates)))
    return candidates.iloc[index]


def _few_shot_from_row(optimizer: Any, instruction: str, row: pd.Series) -> str:
    sample_input = str(row[optimizer.task.x_column])
    sample_target = str(row[optimizer.task.y_column])
    fallback = CAPO_FEWSHOT_TEMPLATE.replace("<input>", sample_input).replace(
        "<output>",
        f"{optimizer.target_begin_marker}{sample_target}{optimizer.target_end_marker}",
    )
    if not optimizer.create_fs_reasoning:
        return fallback
    predictions, sequences = optimizer.predictor.predict([instruction], [sample_input])
    if isinstance(predictions, str):
        predictions = [predictions]
    if isinstance(sequences, str):
        sequences = [sequences]
    if not predictions or not sequences:
        return fallback
    generated = str(sequences[0]).replace(sample_input, "", 1).strip()
    if (
        str(predictions[0]).casefold() == sample_target.casefold()
        or not optimizer.check_fs_accuracy
    ):
        return CAPO_FEWSHOT_TEMPLATE.replace("<input>", sample_input).replace(
            "<output>", generated
        )
    return fallback


def _mutate_examples(
    optimizer: Any,
    child: Prompt,
    instruction: str,
    mode: str,
    diagnostics: Mapping[str, Any],
) -> list[str]:
    examples = list(child.few_shots)
    upper = int(optimizer.upper_shots)
    if mode == "cost":
        if examples:
            examples.pop(random.randrange(len(examples)))
    elif mode == "fairness":
        if len(examples) < upper:
            row = _targeted_row(optimizer, diagnostics)
            if row is not None:
                candidate = _few_shot_from_row(optimizer, instruction, row)
                if candidate not in examples:
                    examples.append(candidate)
    elif mode == "quality":
        if len(examples) < upper:
            examples.extend(
                build_few_shot_examples(
                    instruction=instruction, num_examples=1, optimizer=optimizer
                )
            )
    else:
        draw = random.random()
        if draw < 1.0 / 3.0 and len(examples) < upper:
            examples.extend(
                build_few_shot_examples(
                    instruction=instruction, num_examples=1, optimizer=optimizer
                )
            )
        elif draw < 2.0 / 3.0 and examples:
            examples.pop(random.randrange(len(examples)))
    random.shuffle(examples)
    return examples


def generate_objective_aware_challengers(optimizer: Any) -> list[Prompt]:
    """Generate crossover children and mutate each toward one Pareto objective."""
    children, parent_pairs = _crossover_with_provenance(optimizer)
    modes, probabilities = _normalised_weights(
        getattr(optimizer, "objective_mutation_weights", None)
    )
    chosen_modes = random.choices(modes, weights=probabilities, k=len(children))
    mutation_prompts: list[str] = []
    guide_records: list[dict[str, Any]] = []
    for child, (mother, father), mode in zip(children, parent_pairs, chosen_modes):
        _, record = _guide_for_mode(optimizer, mother, father, mode)
        guide_records.append(record)
        mutation_prompts.append(
            _OBJECTIVE_TEMPLATE.replace("<task>", str(optimizer.task.task_description))
            .replace(
                "<labels>", ", ".join((str(value) for value in optimizer.task.classes))
            )
            .replace("<instruction>", child.instruction)
            .replace("<goal>", _GOALS[mode])
            .replace("<diagnostics>", _compact_diagnostics(record))
        )
    responses = optimizer.meta_llm.get_response(mutation_prompts)
    if isinstance(responses, str):
        responses = [responses]
    challengers: list[Prompt] = []
    for child, response, mode, record in zip(
        children, responses, chosen_modes, guide_records
    ):
        instruction = extract_from_tag(response, "<prompt>", "</prompt>").strip()
        if not instruction:
            instruction = child.instruction
        examples = _mutate_examples(
            optimizer, child, instruction, mode, dict(record.get("diagnostics") or {})
        )
        challengers.append(Prompt(instruction, examples))
    optimizer.last_objective_mutation_modes = list(chosen_modes)
    return challengers
