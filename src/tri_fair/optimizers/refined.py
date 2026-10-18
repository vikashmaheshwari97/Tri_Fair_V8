"""Experimental improvement of Tri-Fair; unchanged measurement and comparison methods."""

from __future__ import annotations
from promptolution.utils.prompt import Prompt
from promptolution.utils.formatting import extract_from_tag
from tri_fair.optimizers.trifair import TriFairOptimizer
from tri_fair.search.contrastive import (
    _anchor_for_mode,
    _diverse_partner,
    _ensure_output_contract,
    _normalise_instruction,
)
from tri_fair.search.refinement import diverse_evidence, refinement_request
from tri_fair.search.mutation import _mutate_examples


class RefinedTriFairOptimizer(TriFairOptimizer):
    method_version = "8.2-experimental-compact-diverse-feedback"

    def _build_seed_population(self):
        # Try demonstrations through measured challengers, rather than paying
        # for random demonstrations during initialization.
        return [
            Prompt(_ensure_output_contract(p.instruction), []) for p in self.prompts
        ]

    def _generate_challengers(self):
        if not self.objective_aware_variation:
            return super()._generate_challengers()
        count = max(0, int(self.challenger_count()))
        if not count or not self.incumbents:
            return []
        modes = self.allocate_modes(count)
        common = list(self._get_common_blocks(self.incumbents) or [])
        anchors, proposals = [], []
        for mode in modes:
            anchor, _ = _anchor_for_mode(self, mode, common)
            partner = _diverse_partner(self, anchor)
            blocks = self.task.get_evaluated_blocks([anchor])[anchor]
            evidence = (
                diverse_evidence(
                    self.task,
                    anchor,
                    blocks,
                    limit=self.feedback_cases,
                )
                if blocks and self.enable_feedback
                else []
            )
            anchors.append(anchor)
            proposals.append(
                refinement_request(self.task, anchor, partner, mode, evidence)
            )
        responses = self._responses(
            self.meta_llm.get_response(proposals), len(proposals)
        )
        output, mapping = [], {}
        existing = set(self.seen_candidates)

        def add(candidate, mode):
            key = candidate.construct_prompt()
            if key in existing:
                return
            existing.add(key)
            output.append(candidate)
            mapping[key] = mode
            self.pending_rewards[key] = mode
            self.mode_attempts[mode] += 1

        for anchor, mode, response in zip(anchors, modes, responses):
            instruction = _normalise_instruction(
                extract_from_tag(response, "<prompt>", "</prompt>")
            )
            if instruction and len(instruction.split()) <= 55:
                # Score the short instruction on actual task examples; no metric
                # bonuses, identity-based labels or post-hoc score adjustments.
                add(Prompt(_ensure_output_contract(instruction), []), mode)
                if (
                    self.enable_demo_search
                    and self.upper_shots
                    and mode in ("quality", "fairness")
                ):
                    record = self.get_fairness_record(anchor) or {}
                    examples = _mutate_examples(
                        self,
                        Prompt(instruction, []),
                        instruction,
                        mode,
                        dict(record.get("diagnostics") or {}),
                    )
                    if examples:
                        add(
                            Prompt(_ensure_output_contract(instruction), examples[:1]),
                            mode,
                        )
            if anchor.few_shots:
                # Exact demo deletion is a measured candidate, never assumed safe.
                add(Prompt(anchor.instruction, []), "cost")
        self.seen_candidates = existing
        self.generated_modes = mapping
        self.operator_history.append(
            {"step": self.current_step, "modes": list(mapping.values())}
        )
        return output
