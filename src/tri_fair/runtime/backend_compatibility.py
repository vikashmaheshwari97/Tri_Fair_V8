"""Check the pinned Pixtral API without importing torch, vLLM or model weights."""

from __future__ import annotations
import argparse
import ast
from importlib import metadata
import json

TRANSFORMERS_PIN = "5.16.1"
VLLM_PIN = "0.24.0"
PIXTRAL_SYMBOLS = frozenset({
    "PixtralRotaryEmbedding", "apply_rotary_pos_emb", "position_ids_in_meshgrid",
})


def check_pixtral_sources(consumer, provider):
    """Verify the actual installed vLLM import contract against Transformers."""
    required = set(PIXTRAL_SYMBOLS)
    for node in ast.walk(ast.parse(consumer)):
        if isinstance(node, ast.ImportFrom) and node.module == "transformers.models.pixtral.modeling_pixtral":
            required.update(alias.name for alias in node.names if alias.name != "*")
    supplied = set()
    for node in ast.parse(provider).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            supplied.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            supplied.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.Assign):
            supplied.update(n.id for target in node.targets for n in ast.walk(target) if isinstance(n, ast.Name))
    return sorted(required - supplied)


def inspect_backend():
    report = {"model": "mistral-3-24b", "gpu_used": False, "errors": []}
    try:
        versions = {name: metadata.version(name) for name in ("vllm", "transformers")}
        report["packages"] = versions
        if versions != {"vllm": VLLM_PIN, "transformers": TRANSFORMERS_PIN}:
            report["errors"].append(
                f"Expected vllm=={VLLM_PIN} and transformers=={TRANSFORMERS_PIN}; found {versions}. "
                "On Rocket, activate the project venv and run: python -m pip install -e '.[gpu]'"
            )
        consumer = metadata.distribution("vllm").locate_file("vllm/model_executor/models/pixtral.py")
        provider = metadata.distribution("transformers").locate_file("transformers/models/pixtral/modeling_pixtral.py")
        missing = check_pixtral_sources(
            consumer.read_text(encoding="utf-8"), provider.read_text(encoding="utf-8")
        )
        report["missing_symbols"] = missing
        if missing:
            report["errors"].append(f"Transformers Pixtral API is missing: {missing}")
    except (metadata.PackageNotFoundError, OSError, SyntaxError) as error:
        report["errors"].append(str(error))
    report["ok"] = not report["errors"]
    return report


def require_mistral_backend():
    result = inspect_backend()
    if not result["ok"]:
        raise RuntimeError("Mistral backend preflight failed: " + "; ".join(result["errors"]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["mistral-3-24b"], required=True)
    parser.parse_args()
    report = inspect_backend()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
