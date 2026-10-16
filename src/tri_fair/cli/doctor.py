"""Check interpreter, installed package and optional GPU backend without loading weights."""

from __future__ import annotations
import argparse
from importlib import metadata
import json
import platform
import sys


def inspect_environment(gpu=False):
    result = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "executable": sys.executable,
        "python_supported": (3, 11) <= sys.version_info[:2] < (3, 13),
    }
    packages = {}
    for name in [
        "tri-fair-research",
        "promptolution",
        "datasets",
        "pymoo",
        "scikit-learn",
        "pytest",
    ]:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    result["packages"] = packages
    result["ok"] = result["python_supported"] and all(
        v is not None for v in packages.values()
    )
    if gpu:
        try:
            import torch
            import vllm

            result["gpu"] = {
                "torch": torch.__version__,
                "vllm": vllm.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "device_count": int(torch.cuda.device_count()),
            }
            result["ok"] = result["ok"] and result["gpu"]["cuda_available"]
        except Exception as error:
            result["gpu"] = {"error": str(error)}
            result["ok"] = False
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    result = inspect_environment(args.gpu)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
