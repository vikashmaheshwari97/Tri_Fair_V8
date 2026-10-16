"""Initialization responsibilities for the current Tri-Fair search."""

from __future__ import annotations
import random
from typing import Optional
import numpy as np
from promptolution.utils.prompt import Prompt
from collections.abc import Mapping, Sequence
from typing import Any
from promptolution.utils.capo_utils import build_few_shot_examples
from promptolution.utils.formatting import extract_from_tag
from tri_fair.search.contrastive import (
    PRIMARY_MODES,
    _ensure_output_contract,
    _normalise_instruction,
    warm_start_prompt,
)

MODES = ("quality", "fairness", "cost", "balanced", "explore")


class ReadyInitializationMixin:
    def _build_initial_population(self) -> list[Prompt]:
        population = self._build_seed_population()
        if self.smart_start_count <= 0 or len(population) < 2:
            return population
        modes = [
            PRIMARY_MODES[index % len(PRIMARY_MODES)]
            for index in range(self.smart_start_count)
        ]
        ordered = sorted(
            population,
            key=lambda prompt: (
                len(prompt.instruction.split()),
                prompt.instruction.casefold(),
            ),
        )
        parent_pairs: list[tuple[Prompt, Prompt]] = []
        prompts: list[str] = []
        for index, mode in enumerate(modes):
            mother = ordered[index % len(ordered)]
            father = ordered[(index * 5 + len(ordered) // 2 + 1) % len(ordered)]
            if father == mother:
                father = ordered[(index + 1) % len(ordered)]
            parent_pairs.append((mother, father))
            prompts.append(
                warm_start_prompt(self, mode=mode, mother=mother, father=father)
            )
        responses = self._responses(self.meta_llm.get_response(prompts), len(prompts))
        existing = {
            prompt.construct_prompt().strip().casefold() for prompt in population
        }
        for mode, response, (mother, father) in zip(modes, responses, parent_pairs):
            instruction = _normalise_instruction(
                extract_from_tag(response, "<prompt>", "</prompt>")
            )
            if not instruction:
                instruction = (
                    mother.instruction.rstrip(" .")
                    + ". Apply the same rule across protected groups and keep the decision concise."
                )
            instruction = _ensure_output_contract(instruction)
            if mode == "cost":
                examples: list[str] = []
            else:
                requested = 2 if mode == "fairness" else 1
                examples = list(
                    build_few_shot_examples(
                        instruction=instruction,
                        num_examples=min(requested, int(self.upper_shots)),
                        optimizer=self,
                    )
                )
            candidate = Prompt(instruction, examples[: int(self.upper_shots)])
            key = candidate.construct_prompt().strip().casefold()
            if not key or key in existing:
                suffix = {
                    "quality": " Use an explicit evidence hierarchy before selecting the label.",
                    "fairness": " Verify that an identity substitution would not change the decision.",
                    "cost": " Use the shortest complete decision rule and output no explanation.",
                    "balanced": " Use a short ordered rule balancing correctness, fairness, and cost.",
                }[mode]
                candidate = Prompt(
                    _ensure_output_contract(mother.instruction.rstrip(" .") + suffix),
                    examples[: int(self.upper_shots)],
                )
                key = candidate.construct_prompt().strip().casefold()
            if key and key not in existing:
                existing.add(key)
                self.smart_start_prompt_keys.add(key)
                population.append(candidate)
        return population

    def _population_diagnostics(
        self, population: Sequence[Prompt], blocks: Sequence[int]
    ) -> dict[str, Any]:
        widest: dict[str, float] = {}
        max_ci_width = 0.0
        for prompt in population:
            record = self._record_for(prompt, blocks)
            diagnostics = dict(record.get("diagnostics") or {})
            raw = diagnostics.get("uncertainty_by_group") or {}
            if isinstance(raw, Mapping):
                for key, value in raw.items():
                    try:
                        number = float(value)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(number):
                        widest[str(key)] = max(widest.get(str(key), 0.0), number)
            try:
                width = float(diagnostics.get("max_ci_width", 0.0))
            except (TypeError, ValueError):
                width = 0.0
            if np.isfinite(width):
                max_ci_width = max(max_ci_width, width)
        return {"uncertainty_by_group": widest, "max_ci_width": max_ci_width}

    @staticmethod
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

    def _build_seed_population(self) -> list[Prompt]:
        population: list[Prompt] = []
        for prompt in self.prompts:
            num_examples = random.randint(0, self.upper_shots)
            few_shots = build_few_shot_examples(
                instruction=prompt.instruction,
                num_examples=num_examples,
                optimizer=self,
            )
            population.append(Prompt(prompt.instruction, few_shots))
        return population

    def _initialize_search_archive(self) -> None:
        self._initialize_ready_population()
        common = self._common_blocks()
        if common:
            self._update_incumbent_front(blocks=common)
        guard = 0
        while len(self.incumbents) + len(self.non_incumbents) > self.population_size:
            before = len(self.incumbents) + len(self.non_incumbents)
            self._select_survivors()
            after = len(self.incumbents) + len(self.non_incumbents)
            guard += 1
            if after >= before or guard > 4 * max(1, before):
                raise RuntimeError(
                    "Tri-Fair initial environmental selection did not converge"
                )
        self.prompts = self.incumbents + self.non_incumbents
        self.scores = self._get_evaluated_vectors(self.prompts).tolist()
        self.last_step_mode = "smart_verified_initialization"

    def _initialize_ready_population(self) -> None:
        self.current_step = 0
        population = self._build_initial_population()
        remaining = set(range(self.task.n_blocks))
        common_blocks: list[int] = []
        vectors: Optional[np.ndarray] = None
        ready: list[bool] = [False] * len(population)
        while remaining:
            diagnostics = (
                self._population_diagnostics(population, common_blocks)
                if common_blocks
                else {}
            )
            block = self.task.select_fairness_block(
                sorted(remaining), current_blocks=common_blocks, diagnostics=diagnostics
            )
            remaining.remove(block)
            common_blocks.append(int(block))
            common_blocks.sort()
            vectors = self._get_block_vectors(population, common_blocks)
            ready = [
                bool(
                    self._record_for(prompt, common_blocks).get("fairness_ready", False)
                )
                for prompt in population
            ]
            if len(common_blocks) >= self.min_initial_ready_blocks and all(ready):
                break
        if vectors is None:
            raise RuntimeError("Tri-Fair could not evaluate the initial population")
        ready_indices = [index for index, value in enumerate(ready) if value]
        if not ready_indices:
            raise RuntimeError(
                "No initial prompt reached statistical fairness readiness even after all development blocks. Check the study manifest and readiness thresholds."
            )
        ready_vectors = vectors[ready_indices]
        fronts = self._non_dominated_sort(ready_vectors)
        first_front = [ready_indices[index] for index in fronts[0]]
        incumbent_set = set(first_front)
        self.incumbents = [population[index] for index in first_front]
        self.non_incumbents = [
            prompt
            for index, prompt in enumerate(population)
            if index not in incumbent_set
        ]
        self.prompts = self.incumbents + self.non_incumbents
        vector_by_prompt = {
            prompt.construct_prompt(): vectors[index]
            for index, prompt in enumerate(population)
        }
        self.scores = [
            vector_by_prompt[prompt.construct_prompt()].tolist()
            for prompt in self.prompts
        ]
        self.initial_common_blocks = list(common_blocks)
        self.last_step_mode = "ready_initialization"
