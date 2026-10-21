"""CPU check of the real v9.4 initialization rendering hook, no model load."""

import json

from promptolution.utils.prompt import Prompt

from tri_fair.cli.preflight_v88 import check as common_check
from tri_fair.optimizers.evidence_racing import EvidenceRacingTriFairOptimizer


def check():
    result = common_check()
    # This invokes the actual method override called during initialization,
    # rather than checking only a standalone rendering helper.
    obj = object.__new__(EvidenceRacingTriFairOptimizer)
    zero = Prompt("Use only stated professional duties. Return a tagged class.")
    demo = "Input: A synthetic professional case.\nOutput: <final_answer>teacher</final_answer>"
    one = obj._build_initial_prompt(zero, [demo])
    if (
        demo not in one.construct_prompt()
        or one.construct_prompt() == zero.construct_prompt()
    ):
        raise RuntimeError("Initial demonstration missing from v9.4 model input")
    result.update(
        initializer_demonstrations_rendered=True,
        release="Tri-Fair-v9.4-experimental",
        quality_improvement_validated=False,
    )
    return result


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
