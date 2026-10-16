"""Check the local Mistral model path without loading weights or using CUDA."""

from __future__ import annotations
import argparse
import json
from tri_fair.config.models import ALL_MODELS
from tri_fair.runtime.models import _resolve_mistral_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["mistral-3-24b"], required=True)
    parser.add_argument("--print-path", action="store_true")
    args = parser.parse_args()
    try:
        path = _resolve_mistral_snapshot(ALL_MODELS[args.model])
    except (OSError, ValueError) as error:
        parser.exit(1, f"Mistral snapshot check failed: {error}\n")
    print(str(path) if args.print_path else json.dumps({
        "model": args.model, "snapshot": str(path), "complete": True,
        "gpu_used": False,
    }, indent=2))


if __name__ == "__main__":
    main()
