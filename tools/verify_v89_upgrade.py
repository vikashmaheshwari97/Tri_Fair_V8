"""Standalone CPU integrity check for the v8.9 overlay and its required base."""

import hashlib
import json
from pathlib import Path


def verify(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "UPGRADE_V89.json").read_text(encoding="utf-8"))
    errors = []
    for name, expected in manifest["upgrade_sha256"].items():
        path = root / name
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            errors.append(name)
    for name, expected in manifest["required_base_git_blob_sha1"].items():
        path = root / name
        if not path.is_file():
            errors.append(name)
            continue
        data = path.read_bytes()
        observed = hashlib.sha1(
            b"blob " + str(len(data)).encode() + b"\0" + data
        ).hexdigest()
        if observed != expected:
            errors.append(name)
    if errors:
        raise RuntimeError(
            "Incomplete or changed upgrade/base files: " + ", ".join(errors)
        )
    return dict(
        ok=True,
        method="Tri-Fair-v8.9",
        gpu_used=False,
        upgrade_files=len(manifest["upgrade_sha256"]),
        verified_base_files=len(manifest["required_base_git_blob_sha1"]),
        gpu_quality_improvement_validated=False,
    )


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
