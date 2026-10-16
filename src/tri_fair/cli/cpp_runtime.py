"""Resolve a loadable Linux C++ runtime before starting GPU worker processes."""

from __future__ import annotations

import argparse
import ctypes.util
import os
from pathlib import Path
import re
import subprocess
import sys

REQUIRED_SYMBOL = b"GLIBCXX_3.4.29"


def candidate_paths():
    """Prefer standard OS/venv libraries, with Rocket compiler paths as fallback."""
    explicit = os.environ.get("TF_LIBSTDCXX_PATH", "").strip()
    if explicit:
        return [Path(explicit).expanduser()]
    directories = [
        Path("/usr/lib64"), Path("/lib64"),
        Path("/usr/lib/x86_64-linux-gnu"), Path("/lib/x86_64-linux-gnu"),
        Path(sys.prefix) / "lib", Path(sys.prefix) / "lib64",
    ]
    directories.extend(
        Path(p) for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p
    )
    paths = [p / "libstdc++.so.6" for p in directories]
    for pattern in [
        "/opt/rh/gcc-toolset-*/root/usr/lib64/libstdc++.so.6",
        "/gpfs/space/software/cluster_software/compilers/gcc-*/lib64/libstdc++.so.6",
    ]:
        import glob

        paths.extend(Path(p) for p in sorted(glob.glob(pattern), reverse=True))
    found = ctypes.util.find_library("stdc++")
    if found and Path(found).is_absolute():
        paths.append(Path(found))
    return paths


def preload_value(library, previous=""):
    """Replace old explicit libstdc++ preloads while retaining unrelated ones."""
    entries = re.split(r"[\s:]+", previous.strip()) if previous.strip() else []
    return ":".join([str(library), *[
        item for item in entries if "libstdc++" not in Path(item).name
    ]])


def resolve_runtime(paths=None, *, probe=None):
    """Check the ABI symbol and verify loading in a fresh interpreter."""
    paths = candidate_paths() if paths is None else paths
    probe = probe_library if probe is None else probe
    checked = []
    seen = set()
    for candidate in paths:
        candidate = Path(candidate).expanduser().resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if any(c.isspace() or c == ":" for c in str(candidate)):
            checked.append(f"{candidate}: cannot use whitespace/colon in LD_PRELOAD path")
            continue
        try:
            data = candidate.read_bytes()
        except OSError as error:
            checked.append(f"{candidate}: {error.strerror}")
            continue
        if REQUIRED_SYMBOL + b"\0" not in data:
            checked.append(f"{candidate}: lacks {REQUIRED_SYMBOL.decode()}")
            continue
        ok, detail = probe(candidate)
        if ok:
            return candidate
        checked.append(f"{candidate}: cannot load: {detail}")
    raise RuntimeError(
        "No loadable libstdc++.so.6 exporting GLIBCXX_3.4.29 was found. "
        "Load an available GCC >=11 runtime module after the Python modules, "
        "or set TF_LIBSTDCXX_PATH to its full library path before sbatch. "
        "No weights have been loaded by this preflight. Checked:\n  "
        + "\n  ".join(checked)
    )


def probe_library(path):
    env = dict(os.environ)
    env["LD_PRELOAD"] = preload_value(path, env.get("LD_PRELOAD", ""))
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import ctypes,sys; ctypes.CDLL(sys.argv[1])", str(path)],
            env=env, capture_output=True, text=True, timeout=20,
        )
        return result.returncode == 0, result.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, str(error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preload-value", action="store_true")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.exit(1, "This helper is for Linux Slurm jobs only.\n")
    try:
        path = resolve_runtime()
    except RuntimeError as error:
        parser.exit(1, f"{error}\n")
    print(preload_value(path, os.environ.get("LD_PRELOAD", "")) if args.preload_value else path)


if __name__ == "__main__":
    main()
