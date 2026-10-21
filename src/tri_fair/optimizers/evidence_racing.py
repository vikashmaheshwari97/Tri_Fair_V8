"""Experimental v9.4: evidence edits plus conservative MO-CAPO-style racing.

Explicit departures from the v9.2 structural reproduction: v9.3 readiness
warm-up, scoped evidence variation, novelty retries, and optional delete-one-
block stability before EARLY rejection. Stability is a deterministic diagnostic,
not a confidence interval, significance test, or guarantee on adaptive search.
"""

from __future__ import annotations

import random
import re

import numpy as np
from promptolution.utils.token_counter import get_token_counter

from tri_fair.optimizers.support_ready import SupportReadyTriFairOptimizer
from tri_fair.search.evidence_v94 import cached_evidence, edit_request
from tri_fair.search.rendering import output_contract, rendered_prompt


class EvidenceRacingTriFairOptimizer(SupportReadyTriFairOptimizer):
    method_version = "9.4-experimental-evidence-and-stable-racing"
    _adaptive_fields = SupportReadyTriFairOptimizer._adaptive_fields + (
        "proposal_round",
        "seen_proposals",
        "proposal_records",
        "edit_outcomes",
    )

    def __init__(
        self,
        *args,
        evidence_variation=True,
        stability_guard=True,
        proposal_retries=8,
        instruction_token_limit=160,
        generation_input_token_limit=2200,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.full_evaluation:
            raise ValueError(
                "v9.4 uses mixed-fidelity racing; use the v9.2 FullEval control separately"
            )
        self.evidence_variation = bool(evidence_variation)
        self.stability_guard = bool(stability_guard)
        self.proposal_retries = int(proposal_retries)
        self.instruction_token_limit = int(instruction_token_limit)
        self.generation_input_token_limit = int(generation_input_token_limit)
        if (
            not 1 <= self.proposal_retries <= 16
            or self.instruction_token_limit < 32
            or self.generation_input_token_limit < 512
        ):
            raise ValueError("Invalid bounded evidence-generation configuration")
        self.meta_token_counter = get_token_counter(self.meta_llm)
        self.proposal_round = 0
        self.seen_proposals = set()
        self.proposal_records = []
        self.edit_outcomes = []

    def _build_initial_prompt(self, prompt, shots):
        return rendered_prompt(
            prompt.instruction, shots, template=prompt.downstream_template
        )

    def _novel(self, child):
        key = child.construct_prompt()
        observed = {k[0] for k in self.runhistory}
        if key in observed or key in self.seen_proposals:
            return False
        self.seen_proposals.add(key)
        return True

    def _generate_challengers(self):
        children = []
        arms = ("quality", "fairness", "compress", "recombine", "capo")
        for attempt in range(self.proposal_retries):
            if len(children) >= self.crossovers_per_iter:
                break
            round_ = self.proposal_round
            self.proposal_round += 1
            arm = arms[round_ % len(arms)] if self.evidence_variation else "capo"
            self.algorithm_counts["proposal_attempts"] += 1
            if arm == "capo":
                for child in super()._generate_challengers():
                    if len(children) < self.crossovers_per_iter and self._novel(child):
                        children.append(child)
                continue
            parent, partner = self._tournament_selection()
            blocks = self._blocks(parent)
            evidence = cached_evidence(self, parent, blocks, offset=round_)
            demos = list(parent.few_shots)
            # A deterministic, actually cheaper candidate: remove ONE complete
            # demonstration unit. WinoBias pro/anti units remain indivisible.
            if arm == "compress" and demos:
                demos = [] if "wino" in self.task.fairness_metric_name else demos[:-1]
                instruction = parent.instruction
                reason = "remove_complete_demonstration_unit"
            else:
                request = edit_request(self, parent, partner, arm, evidence)
                response = self.meta_llm.get_response([request])[0]
                match = re.search(r"<prompt>(.*?)</prompt>", response, re.S | re.I)
                if not match:
                    self.algorithm_counts["invalid_meta_responses"] += 1
                    continue
                instruction = output_contract(match.group(1).strip())
                reason = "scoped_cached_search_edit"
                if self.token_counter(instruction) > self.instruction_token_limit:
                    self.algorithm_counts["overlong_instructions"] += 1
                    continue
                # IDs and verbatim case input cannot become memorized rules.
                if any(
                    c["search_id"] in instruction
                    or (len(c["input"]) > 80 and c["input"] in instruction)
                    for c in evidence["cases"]
                ):
                    self.algorithm_counts["case_copy_rejections"] += 1
                    continue
            child = rendered_prompt(
                instruction, demos, template=parent.downstream_template
            )
            if arm == "compress" and self.token_counter(
                child.construct_prompt()
            ) >= self.token_counter(parent.construct_prompt()):
                self.algorithm_counts["nonreducing_compressions"] += 1
                continue
            if not self._novel(child):
                self.algorithm_counts["duplicate_proposals"] += 1
                continue
            children.append(child)
            self.proposal_records.append(
                dict(
                    key=child.construct_prompt(),
                    parent=parent,
                    arm=arm,
                    blocks=blocks,
                    reason=reason,
                )
            )
            self._event(
                "evidence_proposal",
                arm=arm,
                evidence_blocks=blocks,
                target_component=evidence["target_component"],
                reason=reason,
                instruction_tokens=self.token_counter(instruction),
            )
        if not children:
            self.algorithm_counts["proposal_retries_exhausted"] += 1
            self._event("proposal_retries_exhausted", attempts=self.proposal_retries)
        return children

    def _stable_dominance(self, incumbent, challenger, blocks):
        """Early pruning requires the same ordering on supported deleted blocks.

        All predictions already exist: this invokes no additional model calls.
        Missing support or fewer than three blocks means inconclusive, not zero.
        """
        if not self.stability_guard:
            return True
        if len(blocks) < 3:
            return False
        for dropped in blocks:
            subset = [b for b in blocks if b != dropped]
            if not self._supported([incumbent, challenger], subset):
                return False
            vectors = self._get_block_vectors([incumbent, challenger], subset)
            if not self._is_dominated(vectors[1], vectors[0]):
                return False
        return True

    def _do_intensification(self, challenger):
        # Preserve common-basis acceptance, mixed-fidelity population and partial
        # rejected genetic material; only EARLY pruning gains a stability guard.
        if challenger in self.incumbents:
            return
        if challenger in self.non_incumbents:
            self.non_incumbents.remove(challenger)
        common = sorted(self._get_common_blocks(self.incumbents) or [])
        if not common:
            raise RuntimeError("No common incumbent basis")
        remaining, visited = list(common), []
        previous = np.full(3, -np.inf)
        self.algorithm_counts["challengers"] += 1
        while remaining:
            block = random.choice(remaining)
            remaining.remove(block)
            visited.append(block)
            vectors = self._get_block_vectors(self.incumbents + [challenger], visited)
            current = vectors[-1]
            self.algorithm_counts["challenger_block_visits"] += 1
            if not remaining:
                self.incumbents.append(challenger)
                self._update_incumbent_front(common)
                self.algorithm_counts["common_basis_survivors"] += 1
                self._event("common_basis_survivor", blocks=sorted(visited))
                self._record_edit_outcome(challenger)
                return
            if not self._supported(self.incumbents + [challenger], visited):
                self.algorithm_counts["deferred_race_support"] += 1
                continue
            if not self._is_dominated(previous, current):
                continue
            previous = current.copy()
            nearest = self._get_closest_incumbent(current, vectors[:-1])
            point_dominated = bool(self._is_dominated(current, vectors[nearest]))
            self.algorithm_counts["nearest_comparisons"] += 1
            reject = point_dominated and self._stable_dominance(
                self.incumbents[nearest], challenger, visited
            )
            if point_dominated and not reject:
                self.algorithm_counts["unstable_early_rejections_deferred"] += 1
            self._event(
                "early_rejected" if reject else "race_continue",
                blocks=sorted(visited),
                point_dominated=point_dominated,
                stability_guard=self.stability_guard,
            )
            if reject:
                self.non_incumbents.append(challenger)
                self.algorithm_counts["early_rejections"] += 1
                self._record_edit_outcome(challenger)
                return

    def _record_edit_outcome(self, child):
        match = next(
            (
                r
                for r in reversed(self.proposal_records)
                if r["key"] == child.construct_prompt()
            ),
            None,
        )
        if match is None:
            return
        parent = match["parent"]
        common = sorted(set(self._blocks(parent)) & set(self._blocks(child)))
        if not common:
            return
        before, after = (self._record_for(p, common) for p in (parent, child))
        supported = bool(before["fairness_ready"] and after["fairness_ready"])
        item = dict(
            arm=match["arm"],
            common_blocks=common,
            fairness_ready=supported,
            quality_change=after["quality"] - before["quality"],
            cost_change=after["cost"] - before["cost"],
            unfairness_change=(after["fairness"] - before["fairness"])
            if supported
            else None,
        )
        self.edit_outcomes = (self.edit_outcomes + [item])[-32:]
        self._event("edit_outcome", **item)

    def search_diagnostics(self):
        d = super().search_diagnostics()
        d.update(
            active_method=(
                "Tri-Fair-v9.4-NoEvidence"
                if not self.evidence_variation
                else "Tri-Fair-v9.4-NoStability"
                if not self.stability_guard
                else "Tri-Fair-v9.4"
            ),
            method_version=self.method_version,
            original_capo_variation=not self.evidence_variation,
            evidence_variation=self.evidence_variation,
            conservative_early_pruning=self.stability_guard,
            stability_interpretation="delete-one-search-block sensitivity; not confidence intervals",
            mathematical_thresholds_changed=False,
            holdout_feedback_used=False,
            edit_outcomes=self.edit_outcomes,
        )
        return d

    def optimizer_state_dict(self):
        state = super().optimizer_state_dict()
        state["evidence_configuration"] = [
            self.evidence_variation,
            self.stability_guard,
            self.proposal_retries,
            self.instruction_token_limit,
            self.generation_input_token_limit,
        ]
        return state

    def load_optimizer_state_dict(self, state):
        if (
            state.get("evidence_configuration")
            != self.optimizer_state_dict()["evidence_configuration"]
        ):
            raise ValueError("Evidence-racing configuration differs from checkpoint")
        super().load_optimizer_state_dict(state)
