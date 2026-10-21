"""Plain full-evaluation PO-Fair comparators, with explicit protocol identities.

NSGA-II-PO in MO-CAPO Algorithms 3--5 includes initialization and CAPO
crossover/mutation. These variants remove custom objective-aware editing, not
those required evolutionary operations. Original selectors remain unchanged.
"""

from __future__ import annotations

from tri_fair.optimizers.baselines import NSGAIIIPOFair, RandomSearchPOFair, SPEA2POFair
from tri_fair.optimizers.budget_fill import BudgetFillMixin
from tri_fair.optimizers.nsga2 import NSGA2Fair


class PlainVariationProtocol:
    def __init__(self, *args, objective_aware_variation=False, **kwargs):
        if objective_aware_variation:
            raise ValueError(
                "Plain baselines use original CAPO variation, not objective-aware edits"
            )
        super().__init__(*args, objective_aware_variation=False, **kwargs)

    def search_diagnostics(self):
        return dict(
            super().search_diagnostics(),
            protocol="v93-plain-full-evaluation",
            active_method=self.protocol_name,
            initialization="same frozen instructions and CAPO few-shot initialization",
            variation="independent proposals"
            if isinstance(self, RandomSearchPOFair)
            else "original CAPO crossover and mutation",
            objective_aware_variation=False,
            full_search_evaluation=True,
            progressive_intensification=False,
            mixed_fidelity_population=False,
            weaker_dominance_selection=False,
            incumbent_advancement=False,
            incumbent_bookkeeping="logging of full-evaluation front only; no racing archive",
            inference_cost_and_unfairness_objectives_unchanged=True,
        )


class PlainNSGAII(PlainVariationProtocol, BudgetFillMixin, NSGA2Fair):
    protocol_name = "NSGAII-PO-Fair-Plain"
    method_version = "9.3-plain-capo-full-evaluation-nsgaii"


class PlainNSGAIII(PlainVariationProtocol, BudgetFillMixin, NSGAIIIPOFair):
    protocol_name = "NSGAIII-PO-Fair-Plain"
    method_version = "9.3-plain-capo-full-evaluation-nsgaiii"


class PlainSPEA2(PlainVariationProtocol, BudgetFillMixin, SPEA2POFair):
    protocol_name = "SPEA2-PO-Fair-Plain"
    method_version = "9.3-plain-capo-full-evaluation-spea2"


class PlainRandomSearch(PlainVariationProtocol, BudgetFillMixin, RandomSearchPOFair):
    protocol_name = "RandomSearch-PO-Fair-Plain"
    method_version = "9.3-independent-random-prompt-proposals"


PLAIN_BASELINES = {
    c.protocol_name: c
    for c in [PlainNSGAII, PlainNSGAIII, PlainSPEA2, PlainRandomSearch]
}
