"""Implementation of the ParetoPromptOptimizer (Multi-Objective CAPO) optimizer."""

import random
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from promptolution.utils.callbacks import BaseCallback
    from promptolution.llms.base_llm import BaseLLM
    from promptolution.predictors.base_predictor import BasePredictor
    from promptolution.tasks.base_task import BaseTask
    from promptolution.utils.config import ExperimentConfig
from promptolution.optimizers.base_optimizer import BaseOptimizer
from tri_fair.runtime.budget import BudgetExhausted
from promptolution.tasks.multi_objective_task import MultiObjectiveTask
from promptolution.utils.capo_utils import (
    build_few_shot_examples,
    perform_crossover,
    perform_mutation,
)
from promptolution.utils.logging import get_logger
from promptolution.utils.prompt import Prompt
from promptolution.utils.templates import (
    CAPO_CROSSOVER_TEMPLATE,
    CAPO_MUTATION_TEMPLATE,
)
from promptolution.utils.token_counter import get_token_counter

logger = get_logger(__name__)


class ParetoPromptOptimizer(BaseOptimizer):
    """Multi-objective variant of CAPO with Pareto-based selection."""

    supports_multi_objective = True

    def __init__(
        self,
        predictor: "BasePredictor",
        task: "BaseTask",
        meta_llm: "BaseLLM",
        initial_prompts: Optional[List[str]] = None,
        crossover_template: Optional[str] = None,
        mutation_template: Optional[str] = None,
        crossovers_per_iter: int = 4,
        upper_shots: int = 5,
        cost_per_input_token: float = 1.0,
        cost_per_output_token: float = 1.0,
        check_fs_accuracy: bool = True,
        create_fs_reasoning: bool = True,
        df_few_shots: Optional[pd.DataFrame] = None,
        callbacks: Optional[List["BaseCallback"]] = None,
        config: Optional["ExperimentConfig"] = None,
        random_selection: bool = False,
        no_weaker_dominance: bool = False,
    ) -> None:
        """Initialize the MO-CAPO-RH optimizer.

        Args:
            predictor: The predictor used to evaluate prompt performance.
            task: The task instance containing data and evaluation settings.
            meta_llm: Meta language model for crossover and mutation generation.
            initial_prompts: Starting prompt strings to seed the population.
            crossover_template: Optional meta-prompt template for crossover.
            mutation_template: Optional meta-prompt template for mutation.
            crossovers_per_iter: Number of crossover operations per iteration.
            upper_shots: Maximum number of few-shot examples to attach.
            cost_per_input_token: Weight applied to input token cost for the cost objective.
            cost_per_output_token: Weight applied to output token cost for the cost objective.
            check_fs_accuracy: Whether to verify few-shot correctness before use.
            create_fs_reasoning: Whether to replace few-shots with model reasoning.
            df_few_shots: Optional dataframe providing few-shot examples. If None, will pop 10% of datapoints from task.
            callbacks: Optional list of optimization callbacks.
            config: Optional experiment configuration object.
            random_selection: Whether to use random selection in Mo-CAPO.
            no_weaker_dominance: Whether to use no weaker dominance in Mo-CAPO.
        """
        self.meta_llm = meta_llm
        self.downstream_llm = predictor.llm
        self.crossovers_per_iter = crossovers_per_iter
        self.upper_shots = upper_shots
        self.cost_per_input_token = cost_per_input_token
        self.cost_per_output_token = cost_per_output_token
        self.check_fs_accuracy = check_fs_accuracy
        self.create_fs_reasoning = create_fs_reasoning
        self.random_selection = random_selection
        self.no_weaker_dominance = no_weaker_dominance
        super().__init__(predictor, task, initial_prompts, callbacks, config)
        self.crossover_template = self._initialize_meta_template(
            crossover_template or CAPO_CROSSOVER_TEMPLATE
        )
        self.mutation_template = self._initialize_meta_template(
            mutation_template or CAPO_MUTATION_TEMPLATE
        )
        self.token_counter = get_token_counter(self.downstream_llm)
        self.df_few_shots = (
            df_few_shots if df_few_shots is not None else task.pop_datapoints(frac=0.1)
        )
        self.incumbents: List[Prompt] = []
        self.non_incumbents: List[Prompt] = []
        self.population_size = len(self.prompts)
        self.runhistory: Dict[Tuple[str, int], Tuple[np.ndarray, int]] = {}
        self.global_min_bounds: Optional[np.ndarray] = None
        self.global_max_bounds: Optional[np.ndarray] = None
        if self.task.task_type == "multi":
            self.n_objectives = len(self.task.tasks) + 1
        else:
            self.n_objectives = 2
        if "block" not in self.task.eval_strategy:
            logger.warning(
                f"ℹ️ CAPO requires 'block' in the eval_strategy, but got {self.task.eval_strategy}. Setting eval_strategy to 'sequential_block'."
            )
            self.task.eval_strategy = "sequential_block"
        if hasattr(self.predictor, "begin_marker") and hasattr(
            self.predictor, "end_marker"
        ):
            self.target_begin_marker = self.predictor.begin_marker
            self.target_end_marker = self.predictor.end_marker
        else:
            self.target_begin_marker = ""
            self.target_end_marker = ""

    def _pre_optimization_loop(self) -> None:
        self.current_step = 0
        population: List[Prompt] = []
        for prompt in self.prompts:
            num_examples = random.randint(0, self.upper_shots)
            few_shots = build_few_shot_examples(
                instruction=prompt.instruction,
                num_examples=num_examples,
                optimizer=self,
            )
            population.append(Prompt(prompt.instruction, few_shots))
        random_block = random.choice(range(self.task.n_blocks))
        initial_vectors = self._get_block_vectors(population, random_block)
        self._update_global_bounds(initial_vectors)
        fronts = self._non_dominated_sort(initial_vectors)
        self.incumbents = [population[i] for i in fronts[0]]
        self.non_incumbents = [population[i] for front in fronts[1:] for i in front]
        self.prompts = self.incumbents + self.non_incumbents
        self.scores = initial_vectors.tolist()

    def _generate_challengers(self) -> List[Prompt]:
        """Generate challengers; fairness subclasses may override this operator."""
        offsprings = perform_crossover(self.prompts, self, self._tournament_selection)
        return perform_mutation(offsprings, self)

    def _step(self) -> List[Prompt]:
        self.current_step += 1
        new_challengers = self._generate_challengers()
        completed_challengers = 0
        for challenger in new_challengers:
            try:
                self._do_intensification(challenger)
                completed_challengers += 1
                self._select_survivors()
                self._advance_one_incumbent()
            except BudgetExhausted:
                if completed_challengers == 0:
                    raise
                break
        self.prompts = self.incumbents + self.non_incumbents
        vecs = self._get_evaluated_vectors(self.prompts)
        self.scores = vecs.tolist()
        return self.prompts

    def _do_intensification(self, challenger: Prompt) -> None:
        if challenger in self.incumbents:
            return
        if challenger in self.non_incumbents:
            self.non_incumbents.remove(challenger)
        common_blocks = self._get_common_blocks(self.incumbents)
        remaining_blocks = set(common_blocks)
        challenger_mean: Optional[np.ndarray] = None
        incumbents_mean: Optional[np.ndarray] = None
        t = 0
        old_scores = np.full((self.n_objectives,), -np.inf)
        while remaining_blocks:
            b = random.choice(tuple(remaining_blocks))
            remaining_blocks.remove(b)
            vecs = self._get_block_vectors(self.incumbents + [challenger], b)
            incumbent_block = vecs[:-1]
            challenger_block = vecs[-1]
            t += 1
            if challenger_mean is None:
                challenger_mean = challenger_block.copy()
                incumbents_mean = incumbent_block.copy()
            else:
                challenger_mean += (challenger_block - challenger_mean) / t
                incumbents_mean += (incumbent_block - incumbents_mean) / t
            if not self._is_dominated(old_scores, challenger_mean):
                continue
            old_scores = challenger_mean.copy()
            idx_closest_incumbent = self._get_closest_incumbent(
                challenger_mean, incumbents_mean
            )
            closest_incumbent_mean = incumbents_mean[idx_closest_incumbent]
            if self._is_dominated(challenger_mean, closest_incumbent_mean):
                self.non_incumbents.append(challenger)
                return
        self.incumbents.append(challenger)
        self._update_incumbent_front(blocks=common_blocks)

    def _get_closest_incumbent(
        self, challenger_mean: np.ndarray, incumbents_mean: np.ndarray
    ) -> int:
        """Return the index of the geometrically closest incumbent using global bounds."""
        if challenger_mean.ndim == 1:
            challenger_mean = challenger_mean.reshape(1, -1)
        if incumbents_mean.ndim == 1:
            incumbents_mean = incumbents_mean.reshape(1, -1)
        if self.global_min_bounds is not None and self.global_max_bounds is not None:
            min_b = self.global_min_bounds
            max_b = self.global_max_bounds
        else:
            raise RuntimeError(
                "Global bounds not available - algorithm should have tracked bounds from evaluations"
            )
        rng = max_b - min_b
        rng[rng == 0] = 1.0
        challenger_norm = (challenger_mean - min_b) / rng
        incumbents_norm = (incumbents_mean - min_b) / rng
        dists = np.linalg.norm(incumbents_norm - challenger_norm, axis=1)
        idx = int(np.argmin(dists))
        return idx

    def _update_incumbent_front(self, blocks: List[int]) -> None:
        vecs = self._get_block_vectors(self.incumbents, blocks)
        fronts = self._non_dominated_sort(vecs)
        new_incumbents = [self.incumbents[i] for i in fronts[0]]
        demoted = [self.incumbents[i] for front in fronts[1:] for i in front]
        self.incumbents = new_incumbents
        self.non_incumbents.extend(demoted)

    def _update_global_bounds(self, vectors: np.ndarray) -> None:
        """Update global min/max bounds with new objective vectors from fresh evaluations."""
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if self.global_min_bounds is None:
            self.global_min_bounds = np.min(vectors, axis=0)
            self.global_max_bounds = np.max(vectors, axis=0)
        else:
            current_min = np.min(vectors, axis=0)
            current_max = np.max(vectors, axis=0)
            self.global_min_bounds = np.minimum(self.global_min_bounds, current_min)
            self.global_max_bounds = np.maximum(self.global_max_bounds, current_max)

    def _get_objective_vectors(self, result) -> np.ndarray:
        if isinstance(self.task, MultiObjectiveTask):
            agg_scores = np.stack(result.agg_scores, axis=1)
        else:
            agg_scores = np.atleast_2d(result.agg_scores).T
        agg_input_tokens = np.asarray(result.agg_input_tokens)
        agg_output_tokens = np.asarray(result.agg_output_tokens)
        cost_scalar = (
            self.cost_per_input_token * agg_input_tokens
            + self.cost_per_output_token * agg_output_tokens
        )
        cost_scalar = cost_scalar.reshape(-1, 1)
        return np.hstack([agg_scores, -cost_scalar])

    def _get_block_vectors(
        self, prompts: List[Prompt], block_idx: Union[int, List[int]]
    ) -> np.ndarray:
        """Get objective vectors for prompts on block(s). Caches results.

        Args:
            prompts: List of prompts to evaluate
            block_idx: Single block index, or list of block indices. If list, returns mean over blocks.

        Returns:
            Objective vectors array of shape (n_prompts, n_objectives)
        """
        block_indices = [block_idx] if isinstance(block_idx, int) else block_idx
        if not block_indices:
            raise ValueError("block_idx cannot be empty")
        prompt_strs = [p.construct_prompt() for p in prompts]
        all_block_vectors = []
        for block_idx_int in block_indices:
            cached_vectors = [None] * len(prompts)
            prompts_to_evaluate = []
            eval_indices = []
            for i, prompt_str in enumerate(prompt_strs):
                key = (prompt_str, block_idx_int)
                if key in self.runhistory:
                    cached_vectors[i] = self.runhistory[key][0]
                else:
                    prompts_to_evaluate.append(prompts[i])
                    eval_indices.append(i)
            if prompts_to_evaluate:
                res = self.task.evaluate(
                    prompts_to_evaluate, self.predictor, block_idx=block_idx_int
                )
                new_vectors = self._get_objective_vectors(res)
                self._update_global_bounds(new_vectors)
                for j, (prompt, idx) in enumerate(
                    zip(prompts_to_evaluate, eval_indices)
                ):
                    key = (prompt_strs[idx], block_idx_int)
                    self.runhistory[key] = (new_vectors[j], self.current_step)
                    cached_vectors[idx] = new_vectors[j]
            block_vectors = (
                np.vstack(cached_vectors)
                if len(cached_vectors) > 1
                else cached_vectors[0].reshape(1, -1)
            )
            all_block_vectors.append(block_vectors)
        if len(block_indices) == 1:
            return all_block_vectors[0]
        else:
            return np.mean(np.stack(all_block_vectors, axis=0), axis=0)

    def _get_evaluated_vectors(self, prompts: List[Prompt]) -> np.ndarray:
        """Get mean vectors over all evaluated blocks for each prompt.

        Note: Each prompt may have been evaluated on different blocks.
        Returns mean over each prompt's evaluated blocks.
        """
        result_vectors = []
        for prompt in prompts:
            prompt_str = prompt.construct_prompt()
            evaluated_blocks = [
                block_idx
                for p_str, block_idx in self.runhistory.keys()
                if p_str == prompt_str
            ]
            if not evaluated_blocks:
                raise ValueError(
                    f"Prompt has no evaluated blocks: {prompt_str[:50]}..."
                )
            prompt_mean = self._get_block_vectors([prompt], evaluated_blocks)
            result_vectors.append(prompt_mean[0])
        return (
            np.vstack(result_vectors)
            if len(result_vectors) > 1
            else result_vectors[0].reshape(1, -1)
        )

    def _get_evaluated_blocks(self, prompts: List[Prompt]) -> Dict[Prompt, List[int]]:
        """Get mapping of prompts to their evaluated block indices.

        Uses runhistory as single source of truth.
        """
        str_to_blocks = {}
        for p_str, block_idx in self.runhistory.keys():
            if p_str not in str_to_blocks:
                str_to_blocks[p_str] = []
            str_to_blocks[p_str].append(block_idx)
        blocks_map = {}
        for prompt in prompts:
            prompt_str = prompt.construct_prompt()
            blocks_map[prompt] = str_to_blocks.get(prompt_str, [])
        return blocks_map

    def _advance_one_incumbent(self) -> None:
        if not self.incumbents:
            return
        blocks_map = self._get_evaluated_blocks(self.incumbents)
        eval_counts = [len(blocks_map[k]) for k in self.incumbents]
        min_count = min(eval_counts)
        least = [inc for inc, c in zip(self.incumbents, eval_counts) if c == min_count]
        chosen_inc = random.choice(least)
        union_blocks: set[int] = set()
        for inc in self.incumbents:
            union_blocks |= set(blocks_map[inc])
        chosen_blocks = set(blocks_map[chosen_inc])
        gap_blocks = union_blocks - chosen_blocks
        if gap_blocks:
            b = random.choice(tuple(gap_blocks))
        else:
            all_blocks = set(range(self.task.n_blocks))
            new_blocks = all_blocks - union_blocks
            if not new_blocks:
                return
            b = random.choice(tuple(new_blocks))
        self._get_block_vectors([chosen_inc], b)

    def _select_survivors(self) -> None:
        """Prune population via Pareto logic to enforce size constraints."""
        while len(self.incumbents) + len(self.non_incumbents) > self.population_size:
            if len(self.non_incumbents) > 0:
                blocks_map = self._get_evaluated_blocks(self.non_incumbents)
                block_sets = list(blocks_map.values())
                first_set = block_sets[0]
                is_homogeneous = all((s == first_set for s in block_sets))
                if not is_homogeneous:
                    counts = [len(s) for s in block_sets]
                    min_count = min(counts)
                    candidates = [i for i, c in enumerate(counts) if c == min_count]
                    victim_idx = random.choice(candidates)
                    self.non_incumbents.pop(victim_idx)
                    continue
                vecs = self._get_evaluated_vectors(self.non_incumbents)
                fronts = self._non_dominated_sort(vecs)
                worst_front_indices = fronts[-1]
                worst_front_vecs = vecs[worst_front_indices]
                dists = self._calculate_crowding_distance(worst_front_vecs)
                min_dist = np.min(dists)
                tied_indices = np.where(dists == min_dist)[0]
                local_worst_idx = np.random.choice(tied_indices)
                victim_idx = worst_front_indices[local_worst_idx]
                self.non_incumbents.pop(victim_idx)
                continue
            common_blocks = self._get_common_blocks(self.incumbents)
            vecs = self._get_block_vectors(self.incumbents, common_blocks)
            dists = self._calculate_crowding_distance(vecs)
            victim_idx = int(np.argmin(dists))
            self.incumbents.pop(victim_idx)

    def _get_common_blocks(self, prompts: List[Prompt]) -> List[int] | None:
        """Get the set of block indices that have been evaluated by all given prompts.

        Uses runhistory as single source of truth.
        """
        if not prompts:
            return None
        block_sets = []
        for prompt in prompts:
            prompt_str = prompt.construct_prompt()
            blocks = {
                block_idx
                for p_str, block_idx in self.runhistory.keys()
                if p_str == prompt_str
            }
            block_sets.append(blocks)
        if not block_sets:
            return None
        common = set.intersection(*block_sets)
        return list(common)

    def _select_parent_from_pool(self, selection_pool: List[Prompt]) -> Prompt:
        """Tournament-pick a parent, preferring incumbents and using crowding for ties."""
        p1, p2 = random.sample(selection_pool, 2)
        if self.random_selection:
            return random.choice((p1, p2))
        if p1 in self.incumbents and p2 in self.incumbents:
            return self._pick_incumbent_by_crowding(p1, p2)
        if p1 in self.incumbents:
            return p1
        if p2 in self.incumbents:
            return p2
        if self.no_weaker_dominance:
            return random.choice((p1, p2))
        blocks_map = self._get_evaluated_blocks([p1, p2])
        blocks1 = blocks_map[p1]
        blocks2 = blocks_map[p2]
        if blocks1 == blocks2:
            blocks_map = self._get_evaluated_blocks(self.non_incumbents)
            same_block = [p for p in self.non_incumbents if blocks_map[p] == blocks1]
            if len(same_block) >= 2:
                vecs = self._get_block_vectors(same_block, blocks1)
                fronts = self._non_dominated_sort(vecs)
                idx1 = same_block.index(p1)
                idx2 = same_block.index(p2)
                ranks = {
                    idx: rank for rank, front in enumerate(fronts) for idx in front
                }
                r1 = ranks[idx1]
                r2 = ranks[idx2]
                if r1 < r2:
                    return p1
                if r2 < r1:
                    return p2
                front_indices = fronts[r1]
                front_vecs = vecs[front_indices]
                dists = self._calculate_crowding_distance(front_vecs)
                pos1 = front_indices.index(idx1)
                pos2 = front_indices.index(idx2)
                if dists[pos1] > dists[pos2]:
                    return p1
                if dists[pos2] > dists[pos1]:
                    return p2
        elif set(blocks1).issubset(blocks2) and blocks1:
            vecs = self._get_block_vectors([p1, p2], blocks1)
            if self._is_dominated(vecs[0], vecs[1]):
                return p2
        elif set(blocks2).issubset(blocks1) and blocks2:
            vecs = self._get_block_vectors([p1, p2], blocks2)
            if self._is_dominated(vecs[1], vecs[0]):
                return p1
        return random.choice((p1, p2))

    def _pick_incumbent_by_crowding(self, p1: Prompt, p2: Prompt) -> Prompt:
        """Break incumbent ties using crowding distance over common evaluated blocks."""
        common_blocks = self._get_common_blocks(self.incumbents)
        inc_vectors = self._get_block_vectors(self.incumbents, common_blocks)
        inc_distances = self._calculate_crowding_distance(inc_vectors)
        p1_idx = self.incumbents.index(p1)
        p2_idx = self.incumbents.index(p2)
        if inc_distances[p1_idx] > inc_distances[p2_idx]:
            return p1
        if inc_distances[p2_idx] > inc_distances[p1_idx]:
            return p2
        return random.choice((p1, p2))

    def _tournament_selection(self, parents=None) -> Tuple[Prompt, Prompt]:
        """Pick two distinct parents via tournament selection."""
        selection_pool = self.incumbents + self.non_incumbents
        parent1 = self._select_parent_from_pool(selection_pool)
        parent2 = self._select_parent_from_pool(selection_pool)
        while parent2 == parent1:
            parent2 = self._select_parent_from_pool(selection_pool)
        return (parent1, parent2)

    @staticmethod
    def _non_dominated_sort(obj_vectors: np.ndarray) -> List[List[int]]:
        """Perform fast non-dominated sorting (NSGA-II) in a vectorized manner."""
        n_solutions = obj_vectors.shape[0]
        greater = obj_vectors[:, None, :] > obj_vectors[None, :, :]
        greater_equal = obj_vectors[:, None, :] >= obj_vectors[None, :, :]
        dominates = np.all(greater_equal, axis=2) & np.any(greater, axis=2)
        domination_counts = dominates.sum(axis=0)
        dominated_solutions = [
            list(np.where(dominates[i])[0]) for i in range(n_solutions)
        ]
        fronts: List[List[int]] = [list(np.where(domination_counts == 0)[0])]
        current_front = 0
        while current_front < len(fronts) and len(fronts[current_front]) > 0:
            next_front: List[int] = []
            for i in fronts[current_front]:
                for dominated in dominated_solutions[i]:
                    domination_counts[dominated] -= 1
                    if domination_counts[dominated] == 0:
                        next_front.append(dominated)
            if len(next_front) > 0:
                fronts.append(next_front)
            current_front += 1
        return fronts

    @staticmethod
    def _is_dominated(vec1, vec2):
        """Returns True if vec2 dominates vec1 in a maximize-all setting."""
        return np.all(vec2 >= vec1) and np.any(vec2 > vec1)

    @staticmethod
    def _calculate_crowding_distance(obj_vectors: np.ndarray) -> np.ndarray:
        """Calculate crowding distance for a set of solutions."""
        num_solutions, num_obj = obj_vectors.shape
        if num_solutions <= 2:
            return np.full(num_solutions, float("inf"))
        distances = np.zeros(num_solutions)
        for i in range(num_obj):
            sorted_indices = np.argsort(obj_vectors[:, i])
            distances[sorted_indices[0]] = float("inf")
            distances[sorted_indices[-1]] = float("inf")
            f_min = obj_vectors[sorted_indices[0], i]
            f_max = obj_vectors[sorted_indices[-1], i]
            if f_max == f_min:
                continue
            slice_indices = sorted_indices[1:-1]
            next_vals = obj_vectors[sorted_indices[2:], i]
            prev_vals = obj_vectors[sorted_indices[:-2], i]
            distances[slice_indices] += (next_vals - prev_vals) / (f_max - f_min)
        return distances
