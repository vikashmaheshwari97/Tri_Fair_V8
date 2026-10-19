"""Tri-Fair prompt construction: stored demonstrations must reach the model.

Historical optimizers deliberately keep their original construction behaviour.
These helpers are opt-in for v8.8 and never change reported token accounting.
"""

from __future__ import annotations

import re
from promptolution.utils.prompt import Prompt
from promptolution.utils.templates import (
    DOWNSTREAM_TEMPLATE,
    DOWNSTREAM_TEMPLATE_W_FEWSHOTS,
)

CONTRACT = "Return one allowed label in <final_answer> tags; no explanation."


def output_contract(instruction: str) -> str:
    """Keep an existing tagged output rule; append at most one fallback rule.

    Conditional mappings and explicit class catalogues are task information.
    Unlike v8.7 we do not append another directive merely because the existing
    directive names two labels. Exact duplicate sentences are removed only for
    unconditional output-format prose, never for substantive task sentences.
    """
    sentences = re.split(r"(?<=[.!?])\s+|\n+", str(instruction).strip())
    kept, format_seen = [], set()
    has_format = False
    for sentence in sentences:
        sentence = " ".join(sentence.split())
        if not sentence:
            continue
        is_format = bool(
            re.match(r"^(return|output|respond)\b", sentence, re.I)
            and "<final_answer>" in sentence.lower()
            and not re.search(r"\b(if|when|unless|otherwise)\b", sentence, re.I)
        )
        if sentence == CONTRACT and has_format:
            continue
        if is_format:
            normalized = sentence.casefold()
            if normalized in format_seen:
                continue
            format_seen.add(normalized)
            has_format = True
        kept.append(sentence)
    text = " ".join(kept)
    if not has_format:
        text = (text + " " + CONTRACT).strip()
    return text


def rendered_prompt(instruction, demonstrations=(), *, template=None):
    """Preserve custom scaffolds while enabling the few-shot insertion point."""
    demos = list(demonstrations)
    defaults = {None, DOWNSTREAM_TEMPLATE, DOWNSTREAM_TEMPLATE_W_FEWSHOTS}
    if template in defaults:
        # Let Prompt select the correct upstream zero/few-shot template.
        prompt = Prompt(instruction, demos)
    else:
        if "<instruction>" not in template:
            raise ValueError("Prompt template must contain <instruction>")
        if demos and "<few_shots>" not in template:
            template = template.replace(
                "<instruction>", "<instruction>\n\n<few_shots>", 1
            )
        prompt = Prompt(instruction, demos, downstream_template=template)
    actual = prompt.construct_prompt()
    if any(demo.strip() not in actual for demo in demos):
        raise ValueError("A stored demonstration is absent from the rendered prompt")
    return prompt
