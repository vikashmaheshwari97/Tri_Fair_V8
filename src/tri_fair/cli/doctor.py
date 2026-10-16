"""Check interpreter, installed package and optional GPU backend without loading weights."""

from __future__ import annotations
import argparse
from importlib import metadata
import json
import platform
import sys


def inspect_cuda_devices(torch):
    """Check hardware and execute kernels before any model weights are loaded."""
    devices = []
    errors = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        capability = torch.cuda.get_device_capability(index)
        memory_gib = props.total_memory / (1024 ** 3)
        devices.append({
            "index": index, "name": props.name,
            "compute_capability": list(capability),
            "memory_gib": round(memory_gib, 2),
        })
        if capability < (8, 0):
            errors.append(f"GPU {index}: {props.name} is too old for this research stack; request H200 (h200-141g).")
        if memory_gib < 70:
            errors.append(f"GPU {index}: {memory_gib:.1f} GiB is below this single-GPU model fleet's 70 GiB minimum; request H200.")
    if not devices:
        errors.append("No CUDA GPUs are visible inside this allocation.")
    if not errors:
        try:
            for index in range(len(devices)):
                probe = torch.ones(8, device=f"cuda:{index}")
                (probe + 1).sum().item()
                torch.cuda.synchronize(index)
        except Exception as error:
            errors.append(f"CUDA kernel probe failed: {error}")
    return {"devices": devices, "errors": errors, "kernel_probe_ok": not errors}


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
                **inspect_cuda_devices(torch),
            }
            result["ok"] = (
                result["ok"] and result["gpu"]["cuda_available"]
                and result["gpu"]["kernel_probe_ok"]
            )
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
