"""Current Tri-Fair optimizer; responsibilities composed without historical inheritance."""

from __future__ import annotations
from tri_fair.search.selection import simplex_reference_directions
from typing import Dict, Optional, Tuple
import numpy as np
from tri_fair.optimizers.pareto import ParetoPromptOptimizer
from tri_fair.runtime.checkpointing import ResumableOptimizerMixin
from tri_fair.data.task import FairnessTask
from tri_fair.search.mutation import DEFAULT_MUTATION_WEIGHTS
from collections.abc import Sequence
from collections import Counter
from tri_fair.search.feedback import generate_reflective_challengers
from tri_fair.search.uncertainty import pooled_bootstrap
from tri_fair.search.objectives import PooledObjectiveMixin
from tri_fair.search.initialization import ReadyInitializationMixin
from tri_fair.search.racing import ProgressiveRacingMixin
from tri_fair.search.archive import ReferenceArchiveMixin

METHOD_VERSION = "8.1-clean-pooled-paired-search"


class TriFairOptimizer(
    PooledObjectiveMixin,
    ReadyInitializationMixin,
    ProgressiveRacingMixin,
    ReferenceArchiveMixin,
    ResumableOptimizerMixin,
    ParetoPromptOptimizer,
):
    method_version = "8.1-clean-pooled-paired-search"

    def __init__(
        self,
        *args,
        fixed_objective_bounds: Optional[Sequence[Sequence[float]]] = None,
        objective_aware_variation: bool = True,
        objective_mutation_weights: Optional[Dict[str, float]] = None,
        min_initial_ready_blocks: int = 2,
        min_racing_blocks: int = 2,
        dominance_epsilons: Sequence[float] = (0.01, 0.5, 0.01),
        fidelity_utilization_thresholds: Sequence[float] = (0.0, 0.3, 0.6, 0.85),
        fidelity_block_targets: Sequence[int] = (2, 3, 4, 5),
        max_confirmation_streak: int = 2,
        quality_guard_scale: float = 0.2,
        fairness_guard_scale: float = 0.1,
        smart_start_count: int = 4,
        archive_cap: int = 8,
        verified_parent_blocks: int = 3,
        stagnation_window: int = 3,
        late_phase_crossovers: int = 2,
        critical_phase_crossovers: int = 1,
        exploration_interval: int = 4,
        reference_partitions: int = 4,
        operator_ucb_scale: float = 0.18,
        quality_floor_start: float = 0.055,
        quality_floor_end: float = 0.02,
        parent_pool_cap: int = 10,
        enable_feedback=True,
        enable_demo_search=True,
        uncertainty_resamples=24,
        uncertainty_alpha=0.1,
        feedback_cases=4,
        full_evaluation=False,
        **kwargs,
    ):
        self.fixed_objective_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.objective_aware_variation = bool(objective_aware_variation)
        self.objective_mutation_weights = dict(
            objective_mutation_weights or DEFAULT_MUTATION_WEIGHTS
        )
        super().__init__(*args, **kwargs)
        if not isinstance(self.task, FairnessTask):
            raise TypeError("TriFair requires tri_fair.data.task.FairnessTask")
        self.n_objectives = 3
        self.aggregate_records: Dict[Tuple[str, Tuple[int, ...]], Dict] = {}
        if fixed_objective_bounds is not None:
            bounds = np.asarray(fixed_objective_bounds, dtype=float)
            if bounds.shape != (2, self.n_objectives):
                raise ValueError(
                    "fixed_objective_bounds must have shape (2, 3): lower then upper"
                )
            lower, upper = bounds
            if not np.all(np.isfinite(bounds)) or np.any(upper <= lower):
                raise ValueError(
                    "fixed objective bounds must be finite and strictly ordered"
                )
            self.fixed_objective_bounds = (lower.copy(), upper.copy())
            self.global_min_bounds = lower.copy()
            self.global_max_bounds = upper.copy()
        self.min_initial_ready_blocks = max(1, int(min_initial_ready_blocks))
        self.min_racing_blocks = max(1, int(min_racing_blocks))
        epsilon = np.asarray(tuple(dominance_epsilons), dtype=float).reshape(-1)
        if (
            epsilon.shape != (3,)
            or np.any(~np.isfinite(epsilon))
            or np.any(epsilon < 0)
        ):
            raise ValueError(
                "dominance_epsilons must contain three finite non-negative values"
            )
        self.dominance_epsilons = epsilon
        self.initial_common_blocks: list[int] = []
        self.last_step_mode = "not_started"
        thresholds = np.asarray(tuple(fidelity_utilization_thresholds), dtype=float)
        targets = np.asarray(tuple(fidelity_block_targets), dtype=int)
        if thresholds.ndim != 1 or targets.ndim != 1 or len(thresholds) != len(targets):
            raise ValueError(
                "fidelity thresholds and block targets must be one-dimensional and equal length"
            )
        if len(thresholds) == 0 or thresholds[0] != 0.0:
            raise ValueError("fidelity_utilization_thresholds must begin at 0.0")
        if np.any(~np.isfinite(thresholds)) or np.any(np.diff(thresholds) <= 0):
            raise ValueError(
                "fidelity utilization thresholds must be finite and increasing"
            )
        if thresholds[-1] >= 1.0:
            raise ValueError(
                "the final fidelity utilization threshold must be below 1.0"
            )
        if np.any(targets <= 0) or np.any(np.diff(targets) < 0):
            raise ValueError(
                "fidelity block targets must be positive and non-decreasing"
            )
        self.fidelity_utilization_thresholds = thresholds
        self.fidelity_block_targets = targets
        self.max_confirmation_streak = max(1, int(max_confirmation_streak))
        self.quality_guard_scale = float(quality_guard_scale)
        self.fairness_guard_scale = float(fairness_guard_scale)
        if self.quality_guard_scale < 0 or self.fairness_guard_scale < 0:
            raise ValueError("guard scales must be non-negative")
        self._confirmation_streak = 0
        self.smart_start_count = max(0, int(smart_start_count))
        self.archive_cap = max(4, int(archive_cap))
        self.verified_parent_blocks = max(1, int(verified_parent_blocks))
        self.smart_start_prompt_keys: set[str] = set()
        self.contrastive_modes: dict[str, str] = {}
        self.last_step_mode = "not_started"
        self.stagnation_window = max(1, int(stagnation_window))
        self.late_phase_crossovers = max(1, int(late_phase_crossovers))
        self.critical_phase_crossovers = max(1, int(critical_phase_crossovers))
        self.exploration_interval = max(1, int(exploration_interval))
        self._stagnation_steps = 0
        self._mode_usage: Counter[str] = Counter()
        self.variation_modes: dict[str, str] = {}
        self.reference_directions = simplex_reference_directions(
            int(reference_partitions)
        )
        self.operator_ucb_scale = float(operator_ucb_scale)
        self.quality_floor_start = float(quality_floor_start)
        self.quality_floor_end = float(quality_floor_end)
        if not 0.0 <= self.quality_floor_end <= self.quality_floor_start <= 0.25:
            raise ValueError(
                "quality floor margins must satisfy 0 <= end <= start <= 0.25"
            )
        self.parent_pool_cap = max(4, int(parent_pool_cap))
        self.mode_attempts: Counter[str] = Counter()
        self.mode_successes: Counter[str] = Counter()
        self.candidate_modes: dict[str, str] = {}
        self._archive_signature: np.ndarray | None = None
        self._archive_stagnation_steps = 0
        if uncertainty_resamples not in (0,) and uncertainty_resamples < 8:
            raise ValueError(
                "use zero to disable uncertainty or at least eight replicates"
            )
        if not 0 < uncertainty_alpha < 0.5 or feedback_cases < 1:
            raise ValueError("invalid uncertainty alpha or feedback case count")
        self.enable_feedback = bool(enable_feedback)
        self.enable_demo_search = bool(enable_demo_search)
        self.uncertainty_resamples = int(uncertainty_resamples)
        self.uncertainty_alpha = float(uncertainty_alpha)
        self.feedback_cases = int(feedback_cases)
        self.full_evaluation = bool(full_evaluation)
        self.pending_rewards = {}
        self.seen_candidates = set()
        self.operator_history = []
        self.demo_history = Counter()
        self._bootstrap_cache = {}

    def _generate_challengers(self):
        if not self.objective_aware_variation:
            return ParetoPromptOptimizer._generate_challengers(self)
        children = generate_reflective_challengers(self)
        modes = self.generated_modes
        for child in children:
            key = child.construct_prompt()
            mode = modes[key]
            self.mode_attempts[mode] += 1
            self.pending_rewards[key] = mode
        self.operator_history.append(
            {"step": self.current_step, "modes": list(modes.values())}
        )
        return children

    def _do_intensification(self, challenger):
        if self.full_evaluation:
            blocks = list(range(self.task.n_blocks))
            self._get_block_vectors([challenger], blocks)
            if self._record_for(challenger, blocks)["fairness_ready"]:
                if challenger not in self.incumbents:
                    self.incumbents.append(challenger)
                self._update_incumbent_front(blocks)
            elif challenger not in self.non_incumbents:
                self.non_incumbents.append(challenger)
            return
        self._race_challenger(challenger)

    def _update_incumbent_front(self, blocks):
        self._select_archive_front(blocks)
        for prompt in self.incumbents:
            mode = self.pending_rewards.pop(prompt.construct_prompt(), None)
            if mode:
                self.mode_successes[mode] += 1
        rejected = {p.construct_prompt() for p in self.non_incumbents}
        for key in rejected:
            self.pending_rewards.pop(key, None)

    def _pre_optimization_loop(self):
        if self.full_evaluation:
            self.min_initial_ready_blocks = self.task.n_blocks
        self._initialize_search_archive()
        self.seen_candidates.update((p.construct_prompt() for p in self.prompts))

    def _required_common_blocks(self):
        return (
            self.task.n_blocks
            if self.full_evaluation
            else self._scheduled_common_blocks()
        )

    def _robust_vectors(self, prompts, blocks):
        return self._get_block_vectors(list(prompts), list(blocks))

    def _bootstrap(self, prompts, blocks):
        key = (
            tuple((p.construct_prompt() for p in prompts)),
            tuple(sorted(set(blocks))),
        )
        if key not in self._bootstrap_cache:
            if len(self._bootstrap_cache) >= 64:
                self._bootstrap_cache.clear()
            self._bootstrap_cache[key] = pooled_bootstrap(
                self.task,
                prompts,
                blocks,
                cost_in=self.cost_per_input_token,
                cost_out=self.cost_per_output_token,
                seed=self.task.seed,
                repeats=self.uncertainty_resamples,
            )
        return self._bootstrap_cache[key]

    def _paired_rejection(self, challenger, incumbent, blocks):
        if self.uncertainty_resamples == 0:
            return True
        replicates = self._bootstrap([challenger, incumbent], blocks)
        if not np.isfinite(replicates).all():
            return False
        differences = replicates[:, 1] - replicates[:, 0]
        lower = np.quantile(differences, self.uncertainty_alpha / 3, axis=0)
        return bool(np.all(lower >= 0) and np.any(lower > 0))
