"""Verify upgraded source and unchanged comparators before a new GPU pilot."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(root=ROOT):
    root = Path(root)
    manifest = json.loads((root / "UPGRADE_V94_EVIDENCE_RACING.json").read_text())
    failures = []
    for section in ("sha256", "base_sha256", "protected_sha256"):
        for name, expected in manifest[section].items():
            path = root / name
            if (
                not path.is_file()
                or hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                != expected
            ):
                failures.append(name)
    if failures:
        raise SystemExit(
            "v9.4 verification failed:\n" + "\n".join(sorted(set(failures)))
        )
    print(
        json.dumps(
            dict(
                ok=True,
                release=manifest["release"],
                upgrade_files=len(manifest["sha256"]),
                verified_base_files=len(manifest["base_sha256"]),
                protected_files=len(manifest["protected_sha256"]),
                baseline_implementations_unchanged=True,
                statistical_objectives_unchanged=True,
                gpu_performance_validated=False,
            ),
            indent=2,
        )
    )
    return True


if __name__ == "__main__":
    verify()
