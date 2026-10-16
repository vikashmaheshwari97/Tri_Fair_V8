"""Balanced objective-mode variation used by fallback proposal generation."""

from __future__ import annotations
from collections.abc import Sequence
from typing import Any
from promptolution.utils.formatting import extract_from_tag
from promptolution.utils.prompt import Prompt
from tri_fair.search.mutation import (
    _GOALS,
    _OBJECTIVE_TEMPLATE,
    _compact_diagnostics,
    _crossover_with_provenance,
    _guide_for_mode,
    _mutate_examples,
    generate_objective_aware_challengers,
)

_CORE_MODES: tuple[str, ...] = (
    "quality",
    "fairness",
    "cost",
    "balanced",
    "explore",
    "quality",
    "fairness",
    "balanced",
)


def _mode_schedule(count: int, step: int) -> list[str]:
    if count <= 0:
        return []
    modes = list(_CORE_MODES)
    if count <= len(modes):
        return modes[:count]
    tail = ["explore", "quality", "fairness", "cost", "balanced"]
    offset = int(step) % len(tail)
    rotated = tail[offset:] + tail[:offset]
    while len(modes) < count:
        modes.extend(rotated)
    return modes[:count]


def _responses_as_list(value: Any, expected: int) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, Sequence):
        values = [str(item) for item in value]
    else:
        values = []
    if len(values) < expected:
        values.extend([""] * (expected - len(values)))
    return values[:expected]


def _normalise_instruction(text: str) -> str:
    return " ".join(str(text).strip().split())


def generate_balanced_challengers(optimizer: Any) -> list[Prompt]:
    """Generate one diagnostically guided batch with deterministic objective cover."""
    children, parent_pairs = _crossover_with_provenance(optimizer)
    if not children:
        return []
    modes = _mode_schedule(len(children), getattr(optimizer, "current_step", 0))
    mutation_prompts: list[str] = []
    guide_records: list[dict[str, Any]] = []
    for child, (mother, father), mode in zip(children, parent_pairs, modes):
        _, record = _guide_for_mode(optimizer, mother, father, mode)
        guide_records.append(dict(record or {}))
        mutation_prompts.append(
            _OBJECTIVE_TEMPLATE.replace("<task>", str(optimizer.task.task_description))
            .replace(
                "<labels>", ", ".join((str(value) for value in optimizer.task.classes))
            )
            .replace("<instruction>", child.instruction)
            .replace("<goal>", _GOALS[mode])
            .replace("<diagnostics>", _compact_diagnostics(record))
        )
    responses = _responses_as_list(
        optimizer.meta_llm.get_response(mutation_prompts), len(children)
    )
    existing = {
        _normalise_instruction(prompt.instruction).casefold()
        for prompt in getattr(optimizer, "prompts", [])
    }
    accepted: set[str] = set(existing)
    instructions: list[str] = []
    retry_positions: list[int] = []
    for index, (child, response) in enumerate(zip(children, responses)):
        instruction = _normalise_instruction(
            extract_from_tag(response, "<prompt>", "</prompt>")
        )
        if not instruction:
            instruction = _normalise_instruction(child.instruction)
        key = instruction.casefold()
        if not instruction or key in accepted:
            retry_positions.append(index)
        else:
            accepted.add(key)
        instructions.append(instruction)
    if retry_positions:
        retry_prompts = [
            mutation_prompts[index]
            + "\nThe previous mutation duplicated an existing instruction. Produce a "
            + "materially different decision procedure while preserving the exact output format."
            for index in retry_positions
        ]
        retry_responses = _responses_as_list(
            optimizer.meta_llm.get_response(retry_prompts), len(retry_positions)
        )
        for position, response in zip(retry_positions, retry_responses):
            candidate = _normalise_instruction(
                extract_from_tag(response, "<prompt>", "</prompt>")
            )
            key = candidate.casefold()
            if candidate and key not in accepted:
                instructions[position] = candidate
                accepted.add(key)
    challengers: list[Prompt] = []
    emitted: set[str] = set(existing)
    for child, instruction, mode, record in zip(
        children, instructions, modes, guide_records
    ):
        instruction = instruction or _normalise_instruction(child.instruction)
        examples = _mutate_examples(
            optimizer, child, instruction, mode, dict(record.get("diagnostics") or {})
        )
        candidate = Prompt(instruction, examples)
        key = candidate.construct_prompt().strip().casefold()
        if not key or key in emitted:
            continue
        emitted.add(key)
        challengers.append(candidate)
    if not challengers:
        return generate_objective_aware_challengers(optimizer)
    optimizer.last_objective_mutation_modes = list(modes[: len(challengers)])
    return challengers
