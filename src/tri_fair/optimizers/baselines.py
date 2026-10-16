"""Additional fully evaluated prompt-search comparators.

All use the existing task metrics, immutable splits and budget controller.
SPEA2/NSGA-III adapt their selection rules to discrete LLM-generated prompts;
these are prompt-search variants, not numeric benchmark reproductions.
"""

from __future__ import annotations
from collections import Counter
import random
import numpy as np
from promptolution.utils.prompt import Prompt
from promptolution.utils.formatting import extract_from_tag
from promptolution.utils.capo_utils import build_few_shot_examples
from tri_fair.optimizers.nsga2 import NSGA2Fair
from tri_fair.search.selection import spea2_fitness, spea2_select
from tri_fair.search.feedback import generate_reflective_challengers


class SPEA2POFair(NSGA2Fair):
    method_version = "8.0-spea2-discrete-prompt-variant"

    def _precompute_selection_metrics(self):
        values = self._get_objective_vectors(
            self.task.evaluate(self.prompts, self.predictor, eval_strategy="full")
        )
        self._spea2_fitness, _ = spea2_fitness(values)
        self._population_metrics = {
            i: {"rank": float(f), "crowding_distance": 0.0}
            for i, f in enumerate(self._spea2_fitness)
        }

    def _tournament_select_one(self, parents=None):
        a, b = random.sample(range(len(self.prompts)), 2)
        winner = min([a, b], key=lambda i: self._spea2_fitness[i])
        return self.prompts[winner]

    def _environmental_selection(self, candidates, obj_vectors):
        return [
            candidates[i]
            for i in spea2_select(
                obj_vectors, min(self.population_size, len(candidates))
            )
        ]


class NSGAIIIPOFair(NSGA2Fair):
    method_version = "8.0-nsga3-pymoo-survival-prompt-variant"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival
        from pymoo.util.ref_dirs import get_reference_directions

        partitions = 1
        while (partitions + 2) * (partitions + 3) // 2 <= self.population_size:
            partitions += 1
        directions = get_reference_directions("das-dennis", 3, n_partitions=partitions)
        self.nsga3_survival = ReferenceDirectionSurvival(directions)

    def _tournament_select_one(self, parents=None):
        return random.choice(self.prompts)

    def _environmental_selection(self, candidates, obj_vectors):
        from pymoo.core.population import Population
        from pymoo.core.problem import Problem

        pop = Population.new(
            "F", -np.asarray(obj_vectors), "source_index", np.arange(len(candidates))
        )
        rng = np.random.default_rng(int(np.random.randint(0, 2**31 - 1)))
        survivors = self.nsga3_survival.do(
            Problem(n_var=0, n_obj=3),
            pop,
            n_survive=min(self.population_size, len(candidates)),
            random_state=rng,
        )
        return [candidates[int(i)] for i in survivors.get("source_index")]


class RandomSearchPOFair(NSGA2Fair):
    """Independent LLM proposals from a frozen seed pool; no score-guided parents."""

    method_version = "8.0-independent-llm-random-proposals"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_seed_pool = [p.instruction for p in self.prompts]

    def _generate_challengers(self):
        requests = []
        for _ in range(self.crossovers_per_iter):
            seed = random.choice(self.random_seed_pool)
            requests.append(
                "Write a diverse, concise alternative task instruction. Preserve the tagged final_answer class output. Do not answer the task. Return only <prompt>instruction</prompt>.\nTask: "
                + self.task.task_description
                + "\nSeed: "
                + seed
            )
        responses = self.meta_llm.get_response(requests)
        if isinstance(responses, str):
            responses = [responses]
        if len(responses) != len(requests):
            raise RuntimeError("random proposal count mismatch")
        output = []
        seen = {p.construct_prompt() for p in self.prompts}
        for response in responses:
            instruction = extract_from_tag(response, "<prompt>", "</prompt>").strip()
            if not instruction:
                continue
            demos = build_few_shot_examples(
                instruction, random.randint(0, self.upper_shots), self
            )
            candidate = Prompt(instruction, list(demos))
            if candidate.construct_prompt() not in seen:
                seen.add(candidate.construct_prompt())
                output.append(candidate)
        return output


class ReflectiveNSGAIIPOFair(NSGA2Fair):
    """Matched enhanced proposal generator, full-evaluation NSGA-II survival."""

    method_version = "8.0-nsga2-shared-reflective-demo-operators"

    def __init__(
        self,
        *args,
        enable_feedback=True,
        enable_demo_search=True,
        feedback_cases=4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.enable_feedback = enable_feedback
        self.enable_demo_search = enable_demo_search
        self.feedback_cases = feedback_cases
        self.seen_candidates = set()
        self.demo_history = Counter()

    def challenger_count(self):
        return self.crossovers_per_iter

    def allocate_modes(self, count):
        modes = ["quality", "fairness", "cost", "balanced", "explore"]
        return random.choices(modes, weights=[0.25, 0.35, 0.2, 0.15, 0.05], k=count)

    def _generate_challengers(self):
        if not self.objective_aware_variation:
            return super()._generate_challengers()
        return generate_reflective_challengers(self)
