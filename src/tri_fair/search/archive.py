"""Archive responsibilities for the current Tri-Fair search."""

from __future__ import annotations
import numpy as np
from promptolution.utils.prompt import Prompt
import math

MODES = ("quality", "fairness", "cost", "balanced", "explore")


class ReferenceArchiveMixin:
    def _normalise_vectors(self, vectors: np.ndarray) -> np.ndarray:
        values = np.atleast_2d(np.asarray(vectors, dtype=float))
        if self.fixed_objective_bounds is not None:
            lower, upper = self.fixed_objective_bounds
        else:
            lower = np.nanmin(values, axis=0)
            upper = np.nanmax(values, axis=0)
        span = np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)
        span[~np.isfinite(span) | (span <= 0)] = 1.0
        output = (values - np.asarray(lower, dtype=float)) / span
        return np.clip(output, 0.0, 1.0)

    def _champion_local_indices(self, vectors: np.ndarray) -> list[int]:
        values = np.atleast_2d(np.asarray(vectors, dtype=float))
        normalised = self._normalise_vectors(values)
        quality = normalised[:, 0]
        cost_score = normalised[:, 1]
        fairness_score = normalised[:, 2]
        floor = float(np.nanmax(quality) - self._adaptive_quality_floor_margin())
        eligible = np.flatnonzero(quality >= floor)
        if not len(eligible):
            eligible = np.arange(len(values))
        deficits = 1.0 - normalised
        utopia_distance = np.linalg.norm(deficits, axis=1)
        geometric = np.prod(np.clip(normalised, 1e-09, 1.0), axis=1) ** (1.0 / 3.0)
        champions = [
            int(np.nanargmax(quality)),
            int(np.nanargmax(cost_score)),
            int(np.nanargmax(fairness_score)),
            int(np.nanargmin(utopia_distance)),
            int(np.nanargmax(geometric)),
            int(eligible[np.nanargmax(cost_score[eligible])]),
            int(eligible[np.nanargmax(fairness_score[eligible])]),
            int(
                np.nanargmax(0.55 * quality + 0.25 * cost_score + 0.2 * fairness_score)
            ),
            int(
                np.nanargmax(0.55 * quality + 0.2 * cost_score + 0.25 * fairness_score)
            ),
        ]
        output: list[int] = []
        for index in champions:
            if index not in output:
                output.append(index)
        return output

    def _cap_front(self, vectors: np.ndarray, capacity: int) -> list[int]:
        from tri_fair.search.selection import truncate_archive

        values = np.atleast_2d(np.asarray(vectors, dtype=float))
        return truncate_archive(
            self._normalise_vectors(values), capacity, self.reference_directions
        )

    def parent_pool(self) -> list[Prompt]:
        verified = self._verified_parent_pool()
        if len(verified) <= self.parent_pool_cap:
            return verified
        common = self._common_blocks()
        if not common:
            return verified[: self.parent_pool_cap]
        robust = self._robust_vectors(verified, common)
        selected = self._cap_front(robust, min(self.parent_pool_cap, len(verified)))
        return [verified[index] for index in selected]

    def _metric_family(self) -> str:
        metric = str(getattr(self.task, "fairness_metric_name", ""))
        if metric.startswith("bbq_bias"):
            return "bbq"
        if metric == "civil_equalized_odds":
            return "civil"
        if metric.startswith("bios_tpr_gap"):
            return "bios"
        return "generic"

    def _mode_weights(self) -> dict[str, float]:
        """Structural dataset priorities; no holdout values are consulted."""
        family = self._metric_family()
        weights = {
            "quality": 0.3,
            "fairness": 0.3,
            "cost": 0.18,
            "balanced": 0.17,
            "explore": 0.05,
        }
        if family == "bbq":
            weights.update(
                quality=0.29, fairness=0.35, cost=0.16, balanced=0.15, explore=0.05
            )
        elif family == "civil":
            weights.update(
                quality=0.35, fairness=0.29, cost=0.15, balanced=0.16, explore=0.05
            )
        elif family == "bios":
            weights.update(
                quality=0.35, fairness=0.34, cost=0.12, balanced=0.14, explore=0.05
            )
        if self._stagnation_steps >= self.stagnation_window:
            weights["explore"] += 0.15
            weights["quality"] += 0.05
        total = sum(weights.values())
        return {key: value / total for key, value in weights.items()}

    def allocate_modes(self, count: int) -> list[str]:
        """Override the v6 scheduler with a success-aware UCB portfolio."""
        count = max(0, int(count))
        if count == 0:
            return []
        selected: list[str] = []
        unseen = [m for m in MODES if self.mode_attempts[m] == 0]
        selected.extend(unseen[:count])
        priors = self._front_deficit_weights()
        total_attempts = sum(self.mode_attempts.values()) + 1
        while len(selected) < count:

            def score(mode: str) -> tuple[float, float, int]:
                attempts = self.mode_attempts[mode] + selected.count(mode)
                successes = self.mode_successes[mode]
                posterior = (successes + 1.0) / (attempts + 2.0)
                exploration = self.operator_ucb_scale * math.sqrt(
                    math.log(total_attempts + len(selected) + 1.0) / (attempts + 1.0)
                )
                return (
                    priors[mode] + posterior + exploration,
                    priors[mode],
                    -MODES.index(mode),
                )

            selected.append(max(MODES, key=score))
        if (
            self.current_step > 0
            and self.current_step % self.exploration_interval == 0
            and ("explore" not in selected)
        ):
            selected[-1] = "explore"
        return selected

    def _adaptive_quality_floor_margin(self) -> float:
        utilization = self._budget_utilization()
        return (
            self.quality_floor_start * (1.0 - utilization)
            + self.quality_floor_end * utilization
        )

    def _front_deficit_weights(self) -> dict[str, float]:
        priors = dict(self._mode_weights())
        common = self._common_blocks()
        if self.incumbents and common:
            robust = self._robust_vectors(self.incumbents, common)
            normalised = self._normalise_vectors(robust)
            best = np.nanmax(normalised, axis=0)
            deficits = np.clip(1.0 - best, 0.0, 1.0)
            priors["quality"] += 0.35 * float(deficits[0])
            priors["cost"] += 0.25 * float(deficits[1])
            priors["fairness"] += 0.35 * float(deficits[2])
            priors["balanced"] += 0.15 * float(np.mean(deficits))
        if self._archive_stagnation_steps >= self.stagnation_window:
            priors["explore"] += 0.2
        total = sum(priors.values())
        return {mode: priors[mode] / total for mode in MODES}

    def _update_front_signature(self, robust_vectors: np.ndarray) -> None:
        if not len(robust_vectors):
            return
        normalised = self._normalise_vectors(robust_vectors)
        signature = np.concatenate(
            (
                np.nanmax(normalised, axis=0),
                np.asarray(
                    [float(np.nanmax(np.nanmin(normalised, axis=1)))], dtype=float
                ),
            )
        )
        if self._archive_signature is None:
            self._archive_signature = signature
            self._archive_stagnation_steps = 0
            return
        threshold = np.asarray((0.0015, 0.002, 0.0015, 0.0015), dtype=float)
        if np.any(signature > self._archive_signature + threshold):
            self._archive_signature = np.maximum(self._archive_signature, signature)
            self._archive_stagnation_steps = 0
        else:
            self._archive_stagnation_steps += 1

    def _select_archive_front(self, blocks: list[int]) -> None:
        if not self.incumbents:
            return
        blocks = sorted(set((int(value) for value in blocks)))
        previous = list(self.incumbents)
        robust_vectors = self._robust_vectors(previous, blocks)
        ready_indices = [
            index
            for index, prompt in enumerate(previous)
            if bool(self._record_for(prompt, blocks).get("fairness_ready", False))
        ]
        if not ready_indices:
            self.non_incumbents.extend(previous)
            self.incumbents = []
            return
        ready_vectors = robust_vectors[ready_indices]
        fronts = self._non_dominated_sort(ready_vectors)
        first_ready_local = list(fronts[0])
        first_original = [ready_indices[index] for index in first_ready_local]
        first_vectors = robust_vectors[first_original]
        kept_local = self._cap_front(
            first_vectors, min(self.archive_cap, self.population_size)
        )
        selected_original = {first_original[index] for index in kept_local}
        self.incumbents = [
            prompt
            for index, prompt in enumerate(previous)
            if index in selected_original
        ]
        selected_keys = {
            prompt.construct_prompt().strip().casefold() for prompt in self.incumbents
        }
        for key in selected_keys:
            mode = self.candidate_modes.get(key)
            if mode:
                self.mode_successes[mode] += 1
                self.candidate_modes.pop(key, None)
        demoted = [
            prompt
            for index, prompt in enumerate(previous)
            if index not in selected_original
        ]
        existing = {
            prompt.construct_prompt().strip().casefold()
            for prompt in self.non_incumbents
        }
        for prompt in demoted:
            key = prompt.construct_prompt().strip().casefold()
            if key not in existing:
                existing.add(key)
                self.non_incumbents.append(prompt)
        if self.incumbents:
            selected_positions = [
                index
                for index, prompt in enumerate(previous)
                if prompt.construct_prompt().strip().casefold() in selected_keys
            ]
            self._update_front_signature(robust_vectors[selected_positions])

    def _verified_parent_pool(self) -> list[Prompt]:
        if not self.incumbents:
            return []
        blocks_map = self._get_evaluated_blocks(self.incumbents)
        required = min(
            int(self.task.n_blocks),
            max(self.verified_parent_blocks, self._required_common_blocks()),
        )
        verified: list[Prompt] = []
        for prompt in self.incumbents:
            blocks = sorted(set((int(value) for value in blocks_map[prompt])))
            if len(blocks) < required:
                continue
            try:
                ready = bool(
                    self._record_for(prompt, blocks).get("fairness_ready", False)
                )
            except Exception:
                ready = False
            if ready:
                verified.append(prompt)
        return verified or list(self.incumbents)
