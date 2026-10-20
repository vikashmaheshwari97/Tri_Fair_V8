"""MO-CAPO algorithm structure extended to quality, cost and pooled unfairness.

Reference: mo374z/mo-capo, src/mo_capo.py at 265848bb6e0482ab7ef11028d0b8483f6560e724.
Uses the repository's existing ParetoPromptOptimizer core and CAPO operators.
Fairness-specific departures are recorded in NOTICE_V92.json: unsupported
comparisons defer, and nonlinear metrics aggregate predictions over block unions.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from promptolution.utils.capo_utils import build_few_shot_examples
from promptolution.utils.prompt import Prompt

from tri_fair.data.task import FairnessTask
from tri_fair.optimizers.pareto import ParetoPromptOptimizer
from tri_fair.runtime.budget import BudgetExhausted
from tri_fair.runtime.checkpointing import ResumableOptimizerMixin
from tri_fair.search.objectives import PooledObjectiveMixin


class MOCAPOFairOptimizer(
    PooledObjectiveMixin, ResumableOptimizerMixin, ParetoPromptOptimizer
):
    """Three-objective MO-CAPO with genuinely heterogeneous evaluation levels."""

    method_version = "9.2-mocapo-three-objective-mixed-fidelity"
    _adaptive_fields = (
        "initial_common_blocks",
        "algorithm_counts",
        "recent_events",
        "pending_advancement",
    )

    def __init__(
        self,
        *args,
        full_evaluation=False,
        no_weaker_dominance=False,
        random_selection=False,
        objective_aware_variation=False,
        **kwargs,
    ):
        if objective_aware_variation:
            raise ValueError("v9.2 main method uses original CAPO variation")
        super().__init__(
            *args,
            no_weaker_dominance=no_weaker_dominance,
            random_selection=random_selection,
            **kwargs,
        )
        if not isinstance(self.task, FairnessTask):
            raise TypeError("Tri-Fair v9.2 requires FairnessTask")
        if self.population_size < 2:
            raise ValueError("MO-CAPO requires at least two initial instructions")
        self.n_objectives = 3
        self.current_step = 0
        self.fixed_objective_bounds = None  # observed global min/max, as in MO-CAPO
        self.aggregate_records = {}
        self.full_evaluation = bool(full_evaluation)
        self.initial_common_blocks = []
        self.algorithm_counts = Counter()
        self.recent_events = []
        self.pending_advancement = False
        self.event_directory = None
        self.task.eval_strategy = "sequential_block"

    @staticmethod
    def _key(prompt):
        return prompt.construct_prompt()

    def set_artifact_directory(self, directory):
        self.event_directory = Path(directory)

    def get_fairness_record(self, prompt, blocks=None):
        if blocks is None and prompt in self.incumbents:
            blocks = sorted(self._get_common_blocks(self.incumbents) or [])
        return super().get_fairness_record(prompt, blocks)

    def _event(self, status, **details):
        event = dict(step=self.current_step, status=status, split="search", **details)
        self.recent_events = (self.recent_events + [event])[-128:]
        if self.event_directory is not None:
            directory = Path(self.event_directory)
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / "mocapo_events.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, sort_keys=True) + "\n")

    def _blocks(self, prompt):
        return sorted(self._get_evaluated_blocks([prompt])[prompt])

    def _supported(self, prompts, blocks):
        return all(self._record_for(p, blocks)["fairness_ready"] for p in prompts)

    def _sync(self):
        self.prompts = self.incumbents + self.non_incumbents
        if self.prompts:
            self.scores = self._get_evaluated_vectors(self.prompts).tolist()

    def _pre_optimization_loop(self):
        self.current_step = 0
        population = []
        for prompt in self.prompts:
            shots = build_few_shot_examples(
                instruction=prompt.instruction,
                num_examples=random.randint(0, self.upper_shots),
                optimizer=self,
            )
            population.append(
                Prompt(
                    prompt.instruction,
                    shots,
                    downstream_template=prompt.downstream_template,
                )
            )
        # A single common random block, not fairness-ready/full initialization.
        blocks = (
            list(range(self.task.n_blocks))
            if self.full_evaluation
            else [random.randrange(self.task.n_blocks)]
        )
        self.initial_common_blocks = blocks
        self._get_block_vectors(population, blocks)
        self.incumbents, self.non_incumbents = population, []
        self._update_incumbent_front(blocks)
        self._sync()
        self._event(
            "initialized",
            blocks=blocks,
            population=len(population),
            fairness_ready=self._supported(population, blocks),
        )

    def _update_incumbent_front(self, blocks):
        blocks = sorted(set(blocks))
        if not blocks:
            raise RuntimeError("Incumbents have no common evaluation basis")
        vectors = self._get_block_vectors(self.incumbents, blocks)
        # An unsupported group estimate is not a three-objective dominance claim.
        # Keep provisional candidates until comparable supported evidence exists.
        if not self._supported(self.incumbents, blocks):
            self.algorithm_counts["deferred_front_support"] += 1
            return
        front = set(self._non_dominated_sort(vectors)[0])
        old = self.incumbents
        self.incumbents = [p for i, p in enumerate(old) if i in front]
        self.non_incumbents.extend(p for i, p in enumerate(old) if i not in front)

    def _do_intensification(self, challenger):
        if challenger in self.incumbents:
            self.algorithm_counts["duplicate_incumbents"] += 1
            return
        if challenger in self.non_incumbents:
            self.non_incumbents.remove(challenger)
        if self.full_evaluation:
            all_blocks = list(range(self.task.n_blocks))
            self._get_block_vectors([challenger], all_blocks)
            self.incumbents.append(challenger)
            self._update_incumbent_front(all_blocks)
            self.algorithm_counts["full_evaluation_challengers"] += 1
            self._event("full_evaluation_ablation", blocks=all_blocks)
            return
        common = sorted(self._get_common_blocks(self.incumbents) or [])
        if not common:
            raise RuntimeError("MO-CAPO common-block invariant violated")
        remaining, visited = list(common), []
        previous_comparison = np.full(3, -np.inf)
        self.algorithm_counts["challengers"] += 1
        while remaining:
            block = random.choice(remaining)
            remaining.remove(block)
            visited.append(block)
            vectors = self._get_block_vectors(self.incumbents + [challenger], visited)
            current = vectors[-1]
            self.algorithm_counts["challenger_block_visits"] += 1
            # Algorithm 2 case (b) takes precedence at full COMMON coverage.
            # No requirement to complete the entire search dataset.
            if not remaining:
                self.incumbents.append(challenger)
                self._update_incumbent_front(common)
                self.algorithm_counts["common_basis_survivors"] += 1
                self._event(
                    "common_basis_survivor",
                    blocks=sorted(visited),
                    total_search_blocks=self.task.n_blocks,
                )
                return
            if not self._supported(self.incumbents + [challenger], visited):
                self.algorithm_counts["deferred_race_support"] += 1
                continue
            if not self._is_dominated(previous_comparison, current):
                continue
            previous_comparison = current.copy()
            nearest = self._get_closest_incumbent(current, vectors[:-1])
            reject = bool(self._is_dominated(current, vectors[nearest]))
            self.algorithm_counts["nearest_comparisons"] += 1
            self._event(
                "early_rejected" if reject else "race_continue",
                blocks=sorted(visited),
                candidate=current.tolist(),
                nearest=vectors[nearest].tolist(),
                fairness_ready=True,
            )
            if reject:
                # Rejected challengers retain their genetic material in P_mu.
                self.non_incumbents.append(challenger)
                self.algorithm_counts["early_rejections"] += 1
                return

    def _advance_one_incumbent(self):
        if not self.incumbents or self.full_evaluation:
            return
        mapping = {p: set(self._blocks(p)) for p in self.incumbents}
        least_count = min(map(len, mapping.values()))
        chosen = random.choice(
            [p for p in self.incumbents if len(mapping[p]) == least_count]
        )
        union = set().union(*mapping.values())
        gaps = union - mapping[chosen]
        missing = gaps or (set(range(self.task.n_blocks)) - union)
        if not missing:
            return
        block = random.choice(sorted(missing))
        self._get_block_vectors([chosen], block)
        self.algorithm_counts["incumbent_advances"] += 1
        self.algorithm_counts["gap_advances" if gaps else "new_block_advances"] += 1
        self._event(
            "incumbent_advanced",
            block=block,
            gap_first=bool(gaps),
            evaluation_levels=[len(self._blocks(p)) for p in self.incumbents],
        )

    def _select_survivors(self):
        while len(self.incumbents) + len(self.non_incumbents) > self.population_size:
            if self.non_incumbents:
                sets = [set(self._blocks(p)) for p in self.non_incumbents]
                if any(s != sets[0] for s in sets[1:]):
                    smallest = min(map(len, sets))
                    idx = random.choice(
                        [i for i, s in enumerate(sets) if len(s) == smallest]
                    )
                    self.algorithm_counts["heterogeneous_survivor_removals"] += 1
                else:
                    vectors = self._get_block_vectors(
                        self.non_incumbents, sorted(sets[0])
                    )
                    if not self._supported(self.non_incumbents, sorted(sets[0])):
                        idx = random.randrange(len(self.non_incumbents))
                    else:
                        worst = self._non_dominated_sort(vectors)[-1]
                        distances = self._calculate_crowding_distance(vectors[worst])
                        idx = worst[
                            random.choice(
                                np.flatnonzero(distances == distances.min()).tolist()
                            )
                        ]
                self.non_incumbents.pop(idx)
            else:
                common = sorted(self._get_common_blocks(self.incumbents))
                vectors = self._get_block_vectors(self.incumbents, common)
                if self._supported(self.incumbents, common):
                    distances = self._calculate_crowding_distance(vectors)
                    idx = int(np.argmin(distances))
                else:
                    idx = random.randrange(len(self.incumbents))
                self.incumbents.pop(idx)

    def _pick_incumbent_by_crowding(self, p1, p2):
        common = sorted(self._get_common_blocks(self.incumbents))
        vectors = self._get_block_vectors(self.incumbents, common)
        if not self._supported(self.incumbents, common):
            return random.choice((p1, p2))
        distances = self._calculate_crowding_distance(vectors)
        a, b = (
            distances[self.incumbents.index(p1)],
            distances[self.incumbents.index(p2)],
        )
        return p1 if a > b else p2 if b > a else random.choice((p1, p2))

    def _select_parent_from_pool(self, selection_pool):
        p1, p2 = random.sample(selection_pool, 2)
        if self.random_selection:
            return random.choice((p1, p2))
        if p1 in self.incumbents and p2 in self.incumbents:
            return self._pick_incumbent_by_crowding(p1, p2)
        if p1 in self.incumbents:
            return p1
        if p2 in self.incumbents:
            return p2
        a, b = set(self._blocks(p1)), set(self._blocks(p2))
        if a == b:
            peers = [p for p in self.non_incumbents if set(self._blocks(p)) == a]
            vectors = self._get_block_vectors(peers, sorted(a))
            if self._supported(peers, sorted(a)):
                fronts = self._non_dominated_sort(vectors)
                ranks = {i: r for r, front in enumerate(fronts) for i in front}
                i, j = peers.index(p1), peers.index(p2)
                if ranks[i] != ranks[j]:
                    return p1 if ranks[i] < ranks[j] else p2
                front = fronts[ranks[i]]
                distances = self._calculate_crowding_distance(vectors[front])
                x, y = distances[front.index(i)], distances[front.index(j)]
                if x != y:
                    return p1 if x > y else p2
        elif not self.no_weaker_dominance and (a < b or b < a):
            shared = sorted(a if a < b else b)
            vectors = self._get_block_vectors([p1, p2], shared)
            if self._supported([p1, p2], shared):
                self.algorithm_counts["weaker_dominance_checks"] += 1
                if a < b and self._is_dominated(vectors[0], vectors[1]):
                    return p2
                if b < a and self._is_dominated(vectors[1], vectors[0]):
                    return p1
        return random.choice((p1, p2))

    def _tournament_selection(self, parents=None):
        pool = self.incumbents + self.non_incumbents
        first = self._select_parent_from_pool(pool)
        # Bounded retries prevent a front with a single preferred incumbent from
        # causing the upstream distinct-parent loop to run forever.
        for _ in range(max(4, 2 * len(pool))):
            second = self._select_parent_from_pool(pool)
            if second != first:
                return first, second
        return first, random.choice([p for p in pool if p != first])

    def _step(self):
        self.current_step += 1
        try:
            if self.pending_advancement:
                self._advance_one_incumbent()
                self.pending_advancement = False
            # Inherited plain CAPO crossover + mutation. No adaptive arm selector,
            # fairness-labelled editor, parent archive or deployment-cost guard.
            children = self._generate_challengers()
            for child in children:
                self._do_intensification(child)
                self._select_survivors()
                self.pending_advancement = True
                self._advance_one_incumbent()
                self.pending_advancement = False
            return self.incumbents + self.non_incumbents
        except BudgetExhausted:
            self.algorithm_counts["budget_interruptions"] += 1
            raise
        finally:
            self._sync()

    def search_diagnostics(self):
        return dict(
            active_method=(
                "Tri-Fair-v9.2-FullEval"
                if self.full_evaluation
                else "Tri-Fair-v9.2-NoWeakerDominance"
                if self.no_weaker_dominance
                else "Tri-Fair-v9.2"
            ),
            method_version=self.method_version,
            progressive_intensification=not self.full_evaluation,
            mixed_fidelity_population=not self.full_evaluation,
            weaker_dominance_selection=not self.no_weaker_dominance,
            incumbent_advancement=not self.full_evaluation,
            original_capo_variation=True,
            adaptive_edit_strategies=False,
            initialization="single shared random block"
            if not self.full_evaluation
            else "full search ablation",
            partial_candidates_enter_population=not self.full_evaluation,
            normalization="global observed finite min/max across all three objectives",
            nonlinear_aggregation="pooled predictions on identical block unions",
            unsupported_fairness="defer objective comparisons; retain provisional incumbents",
            incumbent_evaluation_levels=[len(self._blocks(p)) for p in self.incumbents],
            population_evaluation_levels=[len(self._blocks(p)) for p in self.prompts],
            algorithm_counts=dict(self.algorithm_counts),
            recent_events=self.recent_events,
            exact_two_objective_numeric_reproduction=False,
            algorithm_structure="MO-CAPO Algorithms 1/2 with third unfairness objective",
            holdout_feedback_used=False,
        )

    def optimizer_state_dict(self):
        state = super().optimizer_state_dict()
        state["mocapo_configuration"] = [
            self.full_evaluation,
            self.no_weaker_dominance,
            self.random_selection,
            self.crossovers_per_iter,
            self.upper_shots,
            self.cost_per_input_token,
            self.cost_per_output_token,
        ]
        return state

    def load_optimizer_state_dict(self, state):
        if (
            state.get("mocapo_configuration")
            != self.optimizer_state_dict()["mocapo_configuration"]
        ):
            raise ValueError("MO-CAPO checkpoint configuration mismatch")
        super().load_optimizer_state_dict(state)
