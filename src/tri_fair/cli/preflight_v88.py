"""CPU-only checks before spending a GPU allocation on the v8.8 pilot."""

import hashlib
import json
from pathlib import Path
from tri_fair.search.rendering import output_contract, rendered_prompt


def check(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[3]
    manifest = json.loads((root / "configs/v88_baseline_integrity.json").read_text())
    mismatches = [
        name
        for name, expected in manifest["sha256"].items()
        if not (root / name).is_file()
        or hashlib.sha256((root / name).read_bytes()).hexdigest() != expected
    ]
    if mismatches:
        raise RuntimeError(
            "Comparison implementation changed; review before resubmission: "
            + ", ".join(mismatches)
        )
    instruction = "Use grammatical evidence. Return a or b in <final_answer> tags."
    clean = output_contract(instruction)
    if output_contract(clean) != clean or clean.count("<final_answer>") != 1:
        raise RuntimeError("Output contract duplication")
    demo = "Input:\nA synthetic example.\nOutput:\n<final_answer>a</final_answer>"
    zero = rendered_prompt(clean)
    one = rendered_prompt(clean, [demo], template=zero.downstream_template)
    removed = rendered_prompt(clean, [], template=one.downstream_template)
    if (
        demo not in one.construct_prompt()
        or one.construct_prompt() == zero.construct_prompt()
    ):
        raise RuntimeError("Few-shot example did not reach rendered prompt")
    if removed.construct_prompt() != zero.construct_prompt():
        raise RuntimeError("Demonstration removal failed to restore zero-shot prompt")
    return dict(
        ok=True,
        gpu_used=False,
        protected_files=len(manifest["sha256"]),
        demonstrations_rendered=True,
        contract_idempotent=True,
        quality_improvement_validated=False,
    )


def main():
    print(json.dumps(check(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
