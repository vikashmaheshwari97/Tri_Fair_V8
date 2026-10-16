"""Reference-direction and error-cell-aware variation for Tri-Fair v4.

The operator is intentionally specific to the proposed method.  It creates one
candidate for each principal objective direction, selects an appropriate archive
elite as the anchor parent, pairs it with a lexically diverse parent, and applies
a concise diagnostic-guided mutation.  Civil Comments fairness mutations target
the signed confusion-matrix cell rather than using only the largest absolute gap.
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
from tri_fair.search.balanced import generate_balanced_challengers

_PRIMARY_MODES: tuple[str, ...] = ("quality", "fairness", "cost", "balanced")
_EXTRA_MODES: tuple[str, ...] = ("explore", "quality", "fairness", "balanced", "cost")
_GOALS: dict[str, str] = {
    "quality": "Improve predictive accuracy. State a crisp toxicity decision boundary that distinguishes direct abuse, threats, dehumanization, and targeted insults from neutral discussion, quotation, counterspeech, and mere identity mention. Preserve group consistency and use at most 80 words.",
    "fairness": "Reduce the signed identity-specific TPR/FPR disparity in the diagnostics without defaulting to non_toxic or lowering overall recall. Use one identity-invariant rule, do not enumerate demographic groups, and use at most 80 words.",
    "cost": "Compress the decision procedure to at most 45 words. Keep only the minimum rules needed for accurate and identity-invariant toxic/non_toxic classification, require the exact label-only output, and prohibit explanation.",
    "balanced": "Create a short ordered decision rule that jointly improves accuracy, weighted token cost, and equalized-odds fairness. Avoid blanket conservatism and demographic lists. Use at most 70 words and preserve the exact label-only output.",
    "explore": "Produce a materially different but concise classification procedure that can discover a new Pareto region. Keep the rule identity-invariant, operational, and at most 80 words.",
}
_MUTATION_TEMPLATE = "You are improving an instruction used by a frozen language model in a\nthree-objective prompt-optimization experiment.\n\nTask:\n<task>\n\nAllowed output labels:\n<labels>\n\nCurrent crossover instruction:\n<instruction>\n\nSearch direction:\n<goal>\n\nMeasured development diagnostics for the anchor parent:\n<diagnostics>\n\nRequirements:\n- Rewrite the instruction itself; never answer a dataset example.\n- Preserve the exact required output-label format.\n- Use protected identities only to enforce an invariant decision rule, never as predictors.\n- Do not obtain apparent fairness by always choosing one label.\n- Do not include chain-of-thought or a rationale in the downstream answer.\n- Return only the rewritten instruction between <prompt> and </prompt> tags.\n"


def _normalise_instruction(text: object) -> str:
    return " ".join(str(text).strip().split())


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
    output = list(_PRIMARY_MODES[:count])
    offset = int(step) % len(_EXTRA_MODES)
    tail = list(_EXTRA_MODES[offset:] + _EXTRA_MODES[:offset])
    while len(output) < count:
        output.extend(tail)
    return output[:count]


def _prompt_tokens(prompt: Prompt) -> set[str]:
    return set(re.findall("[a-z0-9_]+", prompt.instruction.casefold()))


def _lexical_distance(left: Prompt, right: Prompt) -> float:
    a = _prompt_tokens(left)
    b = _prompt_tokens(right)
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


def _anchor_for_mode(
    optimizer: Any, mode: str, blocks: Sequence[int]
) -> tuple[Prompt, dict[str, Any]]:
    pool = list(
        getattr(optimizer, "incumbents", []) or getattr(optimizer, "prompts", [])
    )
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
        cost_bound = 100.0
        try:
            cost_bound = float(
                optimizer.task.fairness_kwargs.get(
                    "normalization_cost_upper_bound", 100.0
                )
            )
        except Exception:
            pass

        def balanced_loss(item: tuple[Prompt, dict[str, Any]]) -> float:
            record = item[1]
            quality_loss = 1.0 - _number(record, "quality", 0.0)
            cost_loss = _number(record, "cost", cost_bound) / max(cost_bound, 1e-12)
            fairness_loss = _number(record, "fairness", 1.0)
            return max(quality_loss, cost_loss, fairness_loss)

        return min(records, key=balanced_loss)
    return random.choice(records)


def _diverse_partner(optimizer: Any, anchor: Prompt) -> Prompt:
    pool = [
        prompt for prompt in list(getattr(optimizer, "prompts", [])) if prompt != anchor
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


def _civil_error_cell_row(
    optimizer: Any, diagnostics: Mapping[str, Any]
) -> pd.Series | None:
    frame = getattr(optimizer, "df_few_shots", None)
    if frame is None or len(frame) == 0:
        return None
    if str(diagnostics.get("metric", "")) != "civil_equalized_odds":
        return None
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
        return None
    _, identity, gap_name, signed_gap = max(ranked)
    candidates = frame
    if gap_name == "tpr_gap":
        desired_label = "toxic"
        desired_identity_value = 1 if signed_gap < 0 else 0
    else:
        desired_label = "non_toxic"
        desired_identity_value = 1 if signed_gap > 0 else 0
    if identity in candidates.columns:
        identity_values = pd.to_numeric(candidates[identity], errors="coerce").fillna(0)
        candidates = candidates[identity_values.astype(int) == desired_identity_value]
    target_column = getattr(optimizer.task, "y_column", None)
    if target_column in candidates.columns:
        candidates = candidates[
            candidates[target_column].astype(str).str.casefold()
            == desired_label.casefold()
        ]
    if len(candidates) == 0:
        return None
    index = int(np.random.randint(0, len(candidates)))
    return candidates.iloc[index]


def _mutate_examples_v4(
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
    elif mode == "fairness" and len(examples) < upper:
        row = _civil_error_cell_row(optimizer, diagnostics)
        if row is None:
            row = _targeted_row(optimizer, diagnostics)
        if row is not None:
            candidate = _few_shot_from_row(optimizer, instruction, row)
            if candidate not in examples:
                examples.append(candidate)
    elif mode == "quality" and len(examples) < upper:
        examples.extend(
            build_few_shot_examples(
                instruction=instruction, num_examples=1, optimizer=optimizer
            )
        )
    elif mode in {"balanced", "explore"}:
        draw = random.random()
        if draw < 0.5 and len(examples) < upper:
            examples.extend(
                build_few_shot_examples(
                    instruction=instruction, num_examples=1, optimizer=optimizer
                )
            )
        elif examples:
            examples.pop(random.randrange(len(examples)))
    return examples[:upper]


def generate_reference_challengers(optimizer: Any) -> list[Prompt]:
    """Generate reference-direction candidates with explicit archive provenance."""
    count = int(getattr(optimizer, "crossovers_per_iter", 4))
    if count <= 0:
        return []
    modes = _mode_schedule(count, getattr(optimizer, "current_step", 0))
    common_blocks = list(optimizer._get_common_blocks(optimizer.incumbents) or [])
    parent_pairs: list[tuple[Prompt, Prompt]] = []
    guide_records: list[dict[str, Any]] = []
    crossover_prompts: list[str] = []
    inherited_examples: list[list[str]] = []
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
        combined = list(anchor.few_shots) + list(partner.few_shots)
        target_count = min(
            int(optimizer.upper_shots),
            (len(anchor.few_shots) + len(partner.few_shots)) // 2,
        )
        inherited_examples.append(
            random.sample(combined, target_count) if combined and target_count else []
        )
    crossover_responses = _responses(
        optimizer.meta_llm.get_response(crossover_prompts), count
    )
    children: list[Prompt] = []
    for response, (anchor, _), examples in zip(
        crossover_responses, parent_pairs, inherited_examples
    ):
        instruction = _normalise_instruction(
            extract_from_tag(response, "<prompt>", "</prompt>")
        )
        if not instruction:
            instruction = _normalise_instruction(anchor.instruction)
        children.append(Prompt(instruction, examples))
    mutation_prompts: list[str] = []
    for child, mode, record in zip(children, modes, guide_records):
        mutation_prompts.append(
            _MUTATION_TEMPLATE.replace("<task>", str(optimizer.task.task_description))
            .replace(
                "<labels>", ", ".join((str(value) for value in optimizer.task.classes))
            )
            .replace("<instruction>", child.instruction)
            .replace("<goal>", _GOALS[mode])
            .replace("<diagnostics>", _compact_diagnostics(record))
        )
    mutation_responses = _responses(
        optimizer.meta_llm.get_response(mutation_prompts), count
    )
    existing = {
        prompt.construct_prompt().strip().casefold()
        for prompt in list(getattr(optimizer, "prompts", []))
    }
    challengers: list[Prompt] = []
    emitted = set(existing)
    for child, response, mode, record in zip(
        children, mutation_responses, modes, guide_records
    ):
        instruction = _normalise_instruction(
            extract_from_tag(response, "<prompt>", "</prompt>")
        )
        if not instruction:
            instruction = _normalise_instruction(child.instruction)
        examples = _mutate_examples_v4(
            optimizer, child, instruction, mode, dict(record.get("diagnostics") or {})
        )
        candidate = Prompt(instruction, examples)
        key = candidate.construct_prompt().strip().casefold()
        if not key or key in emitted:
            continue
        emitted.add(key)
        challengers.append(candidate)
    if not challengers:
        return generate_balanced_challengers(optimizer)
    optimizer.last_objective_mutation_modes = list(modes[: len(challengers)])
    return challengers
