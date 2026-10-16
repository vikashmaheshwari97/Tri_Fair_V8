"""Dataset-specific, contrastive reference variation for Tri-Fair v5.

V4's reference operator was intentionally tuned to Civil Comments and its goals
mentioned toxicity even when the active dataset was BBQ or Bias-in-Bios.  V5
uses metric-specific goals, preserves quality/cost/fairness/balanced champions,
and creates contrastive few-shot pairs from the frozen few-shot split only.
"""

from __future__ import annotations
import random
import re
from collections.abc import Mapping, Sequence
from typing import Any
import numpy as np
import pandas as pd
from promptolution.utils.capo_utils import build_few_shot_examples
from promptolution.utils.formatting import extract_from_tag
from promptolution.utils.prompt import Prompt
from tri_fair.search.mutation import (
    _compact_diagnostics,
    _few_shot_from_row,
    _targeted_row,
)
from tri_fair.search.reference import generate_reference_challengers

PRIMARY_MODES: tuple[str, ...] = ("quality", "fairness", "cost", "balanced")
EXTRA_MODES: tuple[str, ...] = ("explore", "quality", "fairness", "balanced", "cost")
MODE_WORD_LIMIT = {
    "quality": 85,
    "fairness": 85,
    "cost": 45,
    "balanced": 70,
    "explore": 85,
}
_GENERIC_GOALS = {
    "quality": "Improve predictive accuracy or macro-F1 while preserving the exact label-only output and avoiding new protected-group disparities. Use a crisp operational rule.",
    "fairness": "Reduce the largest measured protected-group disparity without collapsing to one label or sacrificing the task decision boundary. Use one invariant rule.",
    "cost": "Compress the instruction and remove repeated wording. Preserve correctness, fairness, and the exact label-only output while using the fewest necessary words.",
    "balanced": "Create a concise ordered decision procedure that jointly improves quality, recurring weighted token cost, and protected-group fairness.",
    "explore": "Produce a materially different but task-faithful and concise decision procedure that may discover a new Pareto region.",
}
_BBQ_GOALS = {
    "quality": "Improve exact multiple-choice accuracy by resolving entities, relations, negation, and option mapping. Use unknown only when the context is genuinely underdetermined.",
    "fairness": "Reduce category/context bias and preserve disambiguated coverage. Require the same textual evidence for every demographic group and never guess from stereotypes.",
    "cost": "Compress to a short evidence-only rule: use context entailment, select unknown when underdetermined, and return exactly one tagged a, b, or c.",
    "balanced": "Create a concise two-stage rule for ambiguity detection and evidence-based option selection that improves accuracy, cost, and demographic invariance together.",
    "explore": "Develop a different concise entailment procedure that checks all options and remains invariant to demographic identity swaps.",
}
_CIVIL_GOALS = {
    "quality": "Improve toxicity accuracy by distinguishing direct abuse, threats, dehumanisation, and targeted insults from neutral discussion, quotation, reporting, counterspeech, and mere identity mention.",
    "fairness": "Reduce the signed identity-specific TPR/FPR disparity without defaulting to non_toxic or lowering toxic recall. Apply one identity-invariant threshold.",
    "cost": "Compress to a minimal universal toxicity rule. Keep the exact toxic/non_toxic output, prohibit explanation, and avoid demographic lists.",
    "balanced": "Create a short ordered toxicity rule that jointly controls identity-term false positives, detects real abuse, and reduces recurring token cost.",
    "explore": "Produce a materially different concise toxicity procedure that remains invariant to identity substitution while preserving genuine-abuse sensitivity.",
}
_BIOS_GOALS = {
    "quality": "Improve macro-F1 by using explicit occupation titles, duties, credentials, workplaces, and professional activities to distinguish related careers.",
    "fairness": "Reduce profession-specific gender TPR gaps. Make the decision invariant to names and pronouns while preserving profession evidence and minority-class recall.",
    "cost": "Compress to a short evidence hierarchy for principal occupation and require exactly one allowed lower-case label with no explanation.",
    "balanced": "Create a concise evidence hierarchy that improves macro-F1, gender invariance, and weighted prompt cost together.",
    "explore": "Produce a different concise occupation-classification procedure based only on semantic job evidence and invariant to gender cues.",
}
_MUTATION_TEMPLATE = "You are improving an instruction used by a frozen language model in a\nthree-objective prompt-optimization experiment.\n\nTask:\n<task>\n\nAllowed output labels:\n<labels>\n\nCurrent crossover instruction:\n<instruction>\n\nSearch direction:\n<goal>\n\nMeasured development diagnostics for the anchor parent:\n<diagnostics>\n\nRequirements:\n- Rewrite the instruction itself; never answer a dataset example.\n- Preserve the exact required output-label format.\n- Do not use protected attributes as predictors.\n- Do not obtain apparent fairness by always selecting one label.\n- Do not request chain-of-thought or a downstream explanation.\n- Return only the rewritten instruction between <prompt> and </prompt> tags.\n"
_WARM_START_TEMPLATE = "Create one strong seed instruction for a frozen classifier.\n\nTask:\n<task>\n\nAllowed labels:\n<labels>\n\nSeed instruction A:\n<a>\n\nSeed instruction B:\n<b>\n\nDesign objective:\n<goal>\n\nThe new instruction must be concise, operational, protected-group invariant, and\nmust require exactly one allowed label inside <final_answer> tags. Return only the\nnew instruction between <prompt> and </prompt> tags.\n"


def _metric_name(optimizer: Any) -> str:
    return str(getattr(optimizer.task, "fairness_metric_name", ""))


def goal_for_mode(optimizer: Any, mode: str) -> str:
    metric = _metric_name(optimizer)
    if metric.startswith("bbq_bias"):
        return _BBQ_GOALS.get(mode, _GENERIC_GOALS[mode])
    if metric == "civil_equalized_odds":
        return _CIVIL_GOALS.get(mode, _GENERIC_GOALS[mode])
    if metric.startswith("bios_tpr_gap"):
        return _BIOS_GOALS.get(mode, _GENERIC_GOALS[mode])
    return _GENERIC_GOALS[mode]


def warm_start_prompt(
    optimizer: Any, *, mode: str, mother: Prompt, father: Prompt
) -> str:
    return (
        _WARM_START_TEMPLATE.replace("<task>", str(optimizer.task.task_description))
        .replace(
            "<labels>", ", ".join((str(value) for value in optimizer.task.classes))
        )
        .replace("<a>", mother.instruction)
        .replace("<b>", father.instruction)
        .replace("<goal>", goal_for_mode(optimizer, mode))
    )


def _normalise_instruction(text: object) -> str:
    return " ".join(str(text).strip().split())


def _word_count(text: str) -> int:
    return len(re.findall("\\S+", text))


def _ensure_output_contract(text: str) -> str:
    output = _normalise_instruction(text)
    if "<final_answer>" not in output:
        output = (
            output.rstrip(" .")
            + ". Return only one allowed label inside <final_answer> tags."
        )
    return output


def _responses(value: Any, expected: int) -> list[str]:
    if isinstance(value, str):
        output = [value]
    elif isinstance(value, Sequence):
        output = [str(item) for item in value]
    else:
        output = []
    if len(output) < expected:
        output.extend([""] * (expected - len(output)))
    return output[:expected]


def _mode_schedule(count: int, step: int) -> list[str]:
    if count <= 0:
        return []
    output = list(PRIMARY_MODES[:count])
    offset = int(step) % len(EXTRA_MODES)
    tail = list(EXTRA_MODES[offset:] + EXTRA_MODES[:offset])
    while len(output) < count:
        output.extend(tail)
    return output[:count]


def _tokens(prompt: Prompt) -> set[str]:
    return set(re.findall("[a-z0-9_]+", prompt.instruction.casefold()))


def _lexical_distance(left: Prompt, right: Prompt) -> float:
    a, b = (_tokens(left), _tokens(right))
    if not a and (not b):
        return 0.0
    return 1.0 - len(a & b) / max(1, len(a | b))


def _record(optimizer: Any, prompt: Prompt, blocks: Sequence[int]) -> dict[str, Any]:
    try:
        value = optimizer.get_fairness_record(prompt, blocks=blocks)
    except Exception:
        try:
            value = optimizer.get_fairness_record(prompt)
        except Exception:
            return {}
    return dict(value or {})


def _number(record: Mapping[str, Any], key: str, default: float) -> float:
    try:
        value = float(record.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _parent_pool(optimizer: Any) -> list[Prompt]:
    getter = getattr(optimizer, "parent_pool", None)
    if callable(getter):
        pool = list(getter())
        if pool:
            return pool
    return list(
        getattr(optimizer, "incumbents", []) or getattr(optimizer, "prompts", [])
    )


def _anchor_for_mode(
    optimizer: Any, mode: str, blocks: Sequence[int]
) -> tuple[Prompt, dict[str, Any]]:
    pool = _parent_pool(optimizer)
    if not pool:
        raise RuntimeError("Tri-Fair has no parent candidates")
    records = [(prompt, _record(optimizer, prompt, blocks)) for prompt in pool]
    if mode == "quality":
        return max(records, key=lambda item: _number(item[1], "quality", -np.inf))
    if mode == "cost":
        return min(records, key=lambda item: _number(item[1], "cost", np.inf))
    if mode == "fairness":
        return min(records, key=lambda item: _number(item[1], "fairness", np.inf))
    if mode == "balanced":
        cost_bound = float(
            getattr(optimizer.task, "fairness_kwargs", {}).get(
                "normalization_cost_upper_bound", 100.0
            )
        )

        def loss(item: tuple[Prompt, dict[str, Any]]) -> float:
            record = item[1]
            return max(
                1.0 - _number(record, "quality", 0.0),
                _number(record, "cost", cost_bound) / max(cost_bound, 1e-12),
                _number(record, "fairness", 1.0),
            )

        return min(records, key=loss)
    return random.choice(records)


def _diverse_partner(optimizer: Any, anchor: Prompt) -> Prompt:
    pool = [prompt for prompt in _parent_pool(optimizer) if prompt != anchor]
    if not pool:
        pool = [
            prompt
            for prompt in list(getattr(optimizer, "prompts", []))
            if prompt != anchor
        ]
    if not pool:
        return anchor
    distances = np.asarray([_lexical_distance(anchor, prompt) for prompt in pool])
    maximum = float(np.max(distances))
    candidates = [
        prompt
        for prompt, distance in zip(pool, distances)
        if float(distance) == maximum
    ]
    return random.choice(candidates)


def _sample_one(frame: pd.DataFrame) -> pd.Series | None:
    if frame is None or len(frame) == 0:
        return None
    return frame.iloc[int(np.random.randint(0, len(frame)))]


def _civil_contrastive_rows(
    optimizer: Any, diagnostics: Mapping[str, Any]
) -> list[pd.Series]:
    frame = getattr(optimizer, "df_few_shots", None)
    if frame is None or len(frame) == 0:
        return []
    rates = diagnostics.get("rates_by_identity") or {}
    ranked: list[tuple[float, str, str, float]] = []
    for identity, values in rates.items():
        if not isinstance(values, Mapping):
            continue
        for gap_name in ("tpr_gap", "fpr_gap"):
            try:
                gap = float(values.get(gap_name, 0.0))
            except (TypeError, ValueError):
                continue
            if np.isfinite(gap):
                ranked.append((abs(gap), str(identity), gap_name, gap))
    if not ranked:
        return []
    _, identity, gap_name, signed_gap = max(ranked)
    target = "toxic" if gap_name == "tpr_gap" else "non_toxic"
    underperforming = (
        1
        if gap_name == "tpr_gap"
        and signed_gap < 0
        or (gap_name == "fpr_gap" and signed_gap > 0)
        else 0
    )
    rows: list[pd.Series] = []
    for identity_value in (underperforming, 1 - underperforming):
        candidates = frame
        if identity in candidates.columns:
            values = pd.to_numeric(candidates[identity], errors="coerce").fillna(0)
            candidates = candidates[values.astype(int) == identity_value]
        target_column = getattr(optimizer.task, "y_column", None)
        if target_column in candidates.columns:
            candidates = candidates[
                candidates[target_column].astype(str).str.casefold() == target
            ]
        row = _sample_one(candidates)
        if row is not None:
            rows.append(row)
    return rows


def _bios_contrastive_rows(
    optimizer: Any, diagnostics: Mapping[str, Any]
) -> list[pd.Series]:
    frame = getattr(optimizer, "df_few_shots", None)
    if frame is None or len(frame) == 0:
        return []
    gaps = diagnostics.get("tpr_gap_by_profession") or {}
    if not gaps:
        return []
    profession, gap = max(gaps.items(), key=lambda item: abs(float(item[1])))
    underperforming = 1 if float(gap) > 0 else 0
    rows: list[pd.Series] = []
    for gender in (underperforming, 1 - underperforming):
        candidates = frame
        target_column = getattr(optimizer.task, "y_column", None)
        if target_column in candidates.columns:
            candidates = candidates[
                candidates[target_column].astype(str).str.casefold()
                == str(profession).casefold()
            ]
        if "gender" in candidates.columns:
            values = pd.to_numeric(candidates["gender"], errors="coerce")
            candidates = candidates[values.fillna(-1).astype(int) == gender]
        row = _sample_one(candidates)
        if row is not None:
            rows.append(row)
    return rows


def _bbq_contrastive_rows(
    optimizer: Any, diagnostics: Mapping[str, Any]
) -> list[pd.Series]:
    frame = getattr(optimizer, "df_few_shots", None)
    if frame is None or len(frame) == 0:
        return []
    by_category = diagnostics.get("by_category") or {}
    ranked: list[tuple[float, str]] = []
    for category, values in by_category.items():
        if not isinstance(values, Mapping):
            continue
        magnitude = max(
            (abs(float(values.get(context, 0.0))) for context in ("ambig", "disambig")),
            default=0.0,
        )
        ranked.append((magnitude, str(category)))
    if not ranked:
        return []
    _, category = max(ranked)
    rows: list[pd.Series] = []
    for context in ("ambig", "disambig"):
        candidates = frame
        if "category" in candidates.columns:
            candidates = candidates[candidates["category"].astype(str) == category]
        if "context_condition" in candidates.columns:
            candidates = candidates[
                candidates["context_condition"].astype(str).str.casefold() == context
            ]
        row = _sample_one(candidates)
        if row is not None:
            rows.append(row)
    return rows


def _contrastive_rows(
    optimizer: Any, diagnostics: Mapping[str, Any]
) -> list[pd.Series]:
    metric = str(diagnostics.get("metric", _metric_name(optimizer)))
    if metric == "civil_equalized_odds":
        return _civil_contrastive_rows(optimizer, diagnostics)
    if metric.startswith("bios_tpr_gap"):
        return _bios_contrastive_rows(optimizer, diagnostics)
    if metric.startswith("bbq_bias"):
        return _bbq_contrastive_rows(optimizer, diagnostics)
    return []


def _deduplicate_examples(examples: Sequence[str], upper: int) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for example in examples:
        key = " ".join(str(example).split()).casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(str(example))
        if len(output) >= upper:
            break
    return output


def _mutate_examples_v5(
    optimizer: Any,
    child: Prompt,
    instruction: str,
    mode: str,
    diagnostics: Mapping[str, Any],
) -> list[str]:
    upper = int(optimizer.upper_shots)
    examples = list(child.few_shots)
    if mode == "cost":
        return []
    if mode == "fairness":
        targeted = [
            _few_shot_from_row(optimizer, instruction, row)
            for row in _contrastive_rows(optimizer, diagnostics)
        ]
        if not targeted:
            row = _targeted_row(optimizer, diagnostics)
            if row is not None:
                targeted = [_few_shot_from_row(optimizer, instruction, row)]
        return _deduplicate_examples(targeted + examples, upper)
    if mode == "quality":
        additions = build_few_shot_examples(
            instruction=instruction, num_examples=1, optimizer=optimizer
        )
        return _deduplicate_examples(examples + list(additions), upper)
    if mode == "balanced":
        targeted_rows = _contrastive_rows(optimizer, diagnostics)
        additions = [
            _few_shot_from_row(optimizer, instruction, row) for row in targeted_rows[:1]
        ]
        if len(examples) + len(additions) < upper:
            additions.extend(
                build_few_shot_examples(
                    instruction=instruction, num_examples=1, optimizer=optimizer
                )
            )
        return _deduplicate_examples(examples + additions, upper)
    additions = build_few_shot_examples(
        instruction=instruction, num_examples=1, optimizer=optimizer
    )
    return _deduplicate_examples(examples + list(additions), upper)


def _repair_prompt(optimizer: Any, instruction: str, mode: str) -> str:
    limit = MODE_WORD_LIMIT[mode]
    return f"Rewrite the instruction below to be materially distinct, operational, and no more than {limit} words. Preserve the task, protected-group invariance, and the exact single-label <final_answer> output contract. Return only <prompt>...</prompt>.\n\nInstruction:\n{instruction}"


def generate_contrastive_challengers(optimizer: Any) -> list[Prompt]:
    count = int(getattr(optimizer, "crossovers_per_iter", 4))
    if count <= 0:
        return []
    modes = _mode_schedule(count, getattr(optimizer, "current_step", 0))
    common_blocks = list(optimizer._get_common_blocks(optimizer.incumbents) or [])
    parent_pairs: list[tuple[Prompt, Prompt]] = []
    guide_records: list[dict[str, Any]] = []
    crossover_prompts: list[str] = []
    inherited: list[list[str]] = []
    for mode in modes:
        anchor, record = _anchor_for_mode(optimizer, mode, common_blocks)
        partner = _diverse_partner(optimizer, anchor)
        parent_pairs.append((anchor, partner))
        guide_records.append(record)
        crossover_prompts.append(
            optimizer.crossover_template.replace("<mother>", anchor.instruction)
            .replace("<father>", partner.instruction)
            .strip()
        )
        combined = _deduplicate_examples(
            list(anchor.few_shots) + list(partner.few_shots), int(optimizer.upper_shots)
        )
        target_count = min(
            len(combined),
            int(optimizer.upper_shots),
            (len(anchor.few_shots) + len(partner.few_shots)) // 2,
        )
        inherited.append(random.sample(combined, target_count) if target_count else [])
    crossover_responses = _responses(
        optimizer.meta_llm.get_response(crossover_prompts), count
    )
    children: list[Prompt] = []
    for response, (anchor, _), examples in zip(
        crossover_responses, parent_pairs, inherited
    ):
        instruction = _normalise_instruction(
            extract_from_tag(response, "<prompt>", "</prompt>")
        )
        if not instruction:
            instruction = anchor.instruction
        children.append(Prompt(_ensure_output_contract(instruction), examples))
    mutation_prompts = [
        _MUTATION_TEMPLATE.replace("<task>", str(optimizer.task.task_description))
        .replace(
            "<labels>", ", ".join((str(value) for value in optimizer.task.classes))
        )
        .replace("<instruction>", child.instruction)
        .replace("<goal>", goal_for_mode(optimizer, mode))
        .replace("<diagnostics>", _compact_diagnostics(record))
        for child, mode, record in zip(children, modes, guide_records)
    ]
    mutation_responses = _responses(
        optimizer.meta_llm.get_response(mutation_prompts), count
    )
    proposed: list[str] = []
    for child, response in zip(children, mutation_responses):
        instruction = _normalise_instruction(
            extract_from_tag(response, "<prompt>", "</prompt>")
        )
        proposed.append(_ensure_output_contract(instruction or child.instruction))
    existing_instructions = {
        _normalise_instruction(prompt.instruction).casefold()
        for prompt in list(getattr(optimizer, "prompts", []))
    }
    repair_positions = [
        index
        for index, (instruction, mode) in enumerate(zip(proposed, modes))
        if _word_count(instruction) > MODE_WORD_LIMIT[mode]
        or instruction.casefold() in existing_instructions
    ]
    if repair_positions:
        repair_responses = _responses(
            optimizer.meta_llm.get_response(
                [
                    _repair_prompt(optimizer, proposed[index], modes[index])
                    for index in repair_positions
                ]
            ),
            len(repair_positions),
        )
        for index, response in zip(repair_positions, repair_responses):
            repaired = _normalise_instruction(
                extract_from_tag(response, "<prompt>", "</prompt>")
            )
            if repaired:
                proposed[index] = _ensure_output_contract(repaired)
    existing_prompts = {
        prompt.construct_prompt().strip().casefold()
        for prompt in list(getattr(optimizer, "prompts", []))
    }
    challengers: list[Prompt] = []
    emitted = set(existing_prompts)
    mode_by_prompt: dict[str, str] = {}
    for child, instruction, mode, record in zip(
        children, proposed, modes, guide_records
    ):
        examples = _mutate_examples_v5(
            optimizer, child, instruction, mode, dict(record.get("diagnostics") or {})
        )
        candidate = Prompt(instruction, examples)
        key = candidate.construct_prompt().strip().casefold()
        if not key or key in emitted:
            continue
        emitted.add(key)
        challengers.append(candidate)
        mode_by_prompt[key] = mode
    if not challengers:
        return generate_reference_challengers(optimizer)
    optimizer.last_objective_mutation_modes = [
        mode_by_prompt[prompt.construct_prompt().strip().casefold()]
        for prompt in challengers
    ]
    optimizer.contrastive_modes = dict(mode_by_prompt)
    return challengers
