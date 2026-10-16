"""Racing responsibilities for the current Tri-Fair search."""

from __future__ import annotations
import random
import numpy as np
from promptolution.utils.prompt import Prompt
from tri_fair.optimizers.pareto import ParetoPromptOptimizer
import math
from collections.abc import Mapping, Sequence
from typing import Any

MODES = ("quality", "fairness", "cost", "balanced", "explore")


class ProgressiveRacingMixin:
    def _advance_one_incumbent(self) -> None:
        if not self.incumbents:
            return
        blocks_map = self._get_evaluated_blocks(self.incumbents)
        counts = [len(blocks_map[prompt]) for prompt in self.incumbents]
        minimum = min(counts)
        least = [
            prompt for prompt, count in zip(self.incumbents, counts) if count == minimum
        ]
        chosen = random.choice(least)
        union_blocks: set[int] = set()
        for prompt in self.incumbents:
            union_blocks.update(blocks_map[prompt])
        chosen_blocks = list(blocks_map[chosen])
        gap_blocks = union_blocks - set(chosen_blocks)
        if gap_blocks:
            candidates = sorted(gap_blocks)
        else:
            candidates = sorted(set(range(self.task.n_blocks)) - union_blocks)
        if not candidates:
            return
        diagnostics = (
            self._record_for(chosen, chosen_blocks).get("diagnostics", {})
            if chosen_blocks
            else {}
        )
        block = self.task.select_fairness_block(
            candidates, current_blocks=chosen_blocks, diagnostics=diagnostics
        )
        self._get_block_vectors([chosen], block)

    def _n_examples(self, blocks: Sequence[int]) -> int:
        if not blocks:
            return 0
        task_blocks = getattr(self.task, "blocks", None)
        if task_blocks is None:
            return 0
        return int(sum((len(task_blocks[int(block)]) for block in set(blocks))))

    @staticmethod
    def _diagnostic_width(record: Mapping[str, Any]) -> float:
        diagnostics = dict(record.get("diagnostics") or {})
        try:
            value = float(diagnostics.get("max_ci_width", 0.0))
        except (TypeError, ValueError):
            return 0.0
        return value if np.isfinite(value) else 0.0

    def _comparison_epsilon(
        self,
        challenger_record: Mapping[str, Any],
        incumbent_record: Mapping[str, Any],
        blocks: Sequence[int],
    ) -> np.ndarray:
        epsilon = self.dominance_epsilons.copy()
        n_examples = max(1, self._n_examples(blocks))
        epsilon[0] = max(epsilon[0], self.quality_guard_scale / math.sqrt(n_examples))
        width = max(
            self._diagnostic_width(challenger_record),
            self._diagnostic_width(incumbent_record),
        )
        epsilon[2] = max(
            epsilon[2],
            self.fairness_guard_scale * width / math.sqrt(max(1, len(set(blocks)))),
        )
        return epsilon

    def _robustly_dominated(
        self,
        challenger_vector: np.ndarray,
        incumbent_vector: np.ndarray,
        challenger_record: Mapping[str, Any],
        incumbent_record: Mapping[str, Any],
        blocks: Sequence[int],
    ) -> bool:
        delta = np.asarray(incumbent_vector, dtype=float) - np.asarray(
            challenger_vector, dtype=float
        )
        epsilon = self._comparison_epsilon(challenger_record, incumbent_record, blocks)
        no_meaningful_regression = bool(np.all(delta >= -epsilon))
        meaningful_gain = bool(np.any(delta > epsilon))
        return no_meaningful_regression and meaningful_gain

    def _step(self) -> list[Prompt]:
        target = self._required_common_blocks()
        candidate = self._confirmation_candidate_for_target(target)
        utilization = self._budget_utilization()
        final_phase = utilization >= float(self.fidelity_utilization_thresholds[-1])
        if candidate is not None and (
            self._confirmation_streak < self.max_confirmation_streak or final_phase
        ):
            self.current_step += 1
            self._confirmation_streak += 1
            return self._confirmation_step_for_target(target)
        self._confirmation_streak = 0
        self.last_step_mode = "search"
        return ParetoPromptOptimizer._step(self)

    def _budget_utilization(self) -> float:
        controller = getattr(self, "budget_controller", None)
        if controller is None:
            return 0.0
        try:
            value = float(controller.utilization)
        except (TypeError, ValueError, AttributeError):
            return 0.0
        return min(1.0, max(0.0, value)) if np.isfinite(value) else 0.0

    def _common_blocks(self) -> list[int]:
        return sorted(
            (int(value) for value in self._get_common_blocks(self.incumbents) or [])
        )

    def _confirmation_candidate_for_target(
        self, target_count: int
    ) -> tuple[int, Prompt] | None:
        if not self.incumbents:
            return None
        blocks_map = self._get_evaluated_blocks(self.incumbents)
        all_blocks = set(range(self.task.n_blocks))
        common = set(all_blocks)
        for prompt in self.incumbents:
            common &= set(blocks_map[prompt])
        if len(common) >= min(target_count, self.task.n_blocks):
            return None
        remaining = sorted(all_blocks - common)
        if not remaining:
            return None
        coverage = {
            block: sum((block in set(blocks_map[prompt]) for prompt in self.incumbents))
            for block in remaining
        }
        best_coverage = max(coverage.values())
        partially_shared = [
            block
            for block, count in coverage.items()
            if count == best_coverage and count > 0
        ]
        if partially_shared:
            target_block = min(partially_shared)
        else:
            target_block = int(
                self.task.select_fairness_block(
                    remaining, current_blocks=sorted(common), diagnostics={}
                )
            )
        missing = [
            prompt
            for prompt in self.incumbents
            if target_block not in set(blocks_map[prompt])
        ]
        if not missing:
            return None

        def priority(prompt: Prompt) -> tuple[float, int, str]:
            prompt_blocks = blocks_map[prompt]
            if prompt_blocks:
                record = self._record_for(prompt, prompt_blocks)
                width = self._diagnostic_width(record)
            else:
                width = float("inf")
            return (-width, len(prompt_blocks), prompt.construct_prompt())

        chosen = sorted(missing, key=priority)[0]
        return (target_block, chosen)

    def _confirmation_step_for_target(self, target_count: int) -> list[Prompt]:
        candidate = self._confirmation_candidate_for_target(target_count)
        if candidate is None:
            self._confirmation_streak = 0
            return ParetoPromptOptimizer._step(self)
        target_block, prompt = candidate
        self._get_block_vectors([prompt], target_block)
        common = self._common_blocks()
        if common:
            self._update_incumbent_front(blocks=common)
        self.prompts = self.incumbents + self.non_incumbents
        self.scores = self._get_evaluated_vectors(self.prompts).tolist()
        self.last_step_mode = "progressive_archive_confirmation"
        return self.prompts

    def challenger_count(self) -> int:
        """Shrink the final batches so strict atomic budgeting wastes less."""
        base = max(1, int(self.crossovers_per_iter))
        utilization = self._budget_utilization()
        if utilization >= 0.96:
            return min(base, self.critical_phase_crossovers)
        if utilization >= 0.88:
            return min(base, self.late_phase_crossovers)
        if utilization >= 0.72:
            return min(base, 4)
        return base

    def _race_challenger(self, challenger: Prompt) -> None:
        if challenger in self.incumbents:
            return
        if challenger in self.non_incumbents:
            self.non_incumbents.remove(challenger)
        common_blocks = self._common_blocks()
        if not common_blocks:
            common_blocks = [
                int(
                    self.task.select_fairness_block(
                        list(range(self.task.n_blocks)),
                        current_blocks=(),
                        diagnostics={},
                    )
                )
            ]
        remaining = set(common_blocks)
        challenger_blocks: list[int] = []
        comparison_floor = min(
            len(common_blocks),
            max(self.min_racing_blocks, self._required_common_blocks()),
        )
        while remaining:
            diagnostics = (
                self._record_for(challenger, challenger_blocks).get("diagnostics", {})
                if challenger_blocks
                else {}
            )
            block = int(
                self.task.select_fairness_block(
                    sorted(remaining),
                    current_blocks=challenger_blocks,
                    diagnostics=diagnostics,
                )
            )
            remaining.remove(block)
            challenger_blocks.append(block)
            challenger_blocks.sort()
            challenger_vector = self._get_block_vectors(
                [challenger], challenger_blocks
            )[0]
            challenger_record = self._record_for(challenger, challenger_blocks)
            if not challenger_record.get("fairness_ready", False):
                continue
            if len(challenger_blocks) < comparison_floor:
                continue
            ready_incumbents, incumbent_vectors = self._ready_incumbent_vectors(
                self.incumbents, challenger_blocks
            )
            if not ready_incumbents:
                continue
            for incumbent, incumbent_vector in zip(ready_incumbents, incumbent_vectors):
                if not self._is_dominated(challenger_vector, incumbent_vector):
                    continue
                incumbent_record = self._record_for(incumbent, challenger_blocks)
                if self._robustly_dominated(
                    challenger_vector,
                    incumbent_vector,
                    challenger_record,
                    incumbent_record,
                    challenger_blocks,
                ) and (
                    not hasattr(self, "_paired_rejection")
                    or self._paired_rejection(challenger, incumbent, challenger_blocks)
                ):
                    self.non_incumbents.append(challenger)
                    return
        final_record = self._record_for(challenger, common_blocks)
        if not final_record.get("fairness_ready", False):
            self.non_incumbents.append(challenger)
            return
        self.incumbents.append(challenger)
        self._update_incumbent_front(blocks=common_blocks)

    def _scheduled_common_blocks(self) -> int:
        utilization = self._budget_utilization()
        index = int(
            np.searchsorted(
                self.fidelity_utilization_thresholds, utilization, side="right"
            )
            - 1
        )
        index = max(0, min(index, len(self.fidelity_block_targets) - 1))
        return min(int(self.task.n_blocks), int(self.fidelity_block_targets[index]))
