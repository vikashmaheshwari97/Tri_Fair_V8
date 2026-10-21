"""Fail-closed source verification for Tri-Fair v9.3; no GPU or network use."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(root=ROOT):
    root = Path(root)
    manifest = json.loads(
        (root / "UPGRADE_V93_PLAIN_BASELINES_SUPPORT.json").read_text(encoding="utf-8")
    )
    failures = []
    for section in ["sha256", "base_sha256", "protected_sha256"]:
        for name, expected in manifest[section].items():
            path = root / name
            if not path.is_file():
                failures.append(f"{name} (missing)")
            elif (
                hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                != expected
            ):
                failures.append(f"{name} ({section}: changed)")
    if failures:
        raise SystemExit(
            "v9.3 verification failed:\n" + "\n".join(sorted(set(failures)))
        )
    print(
        json.dumps(
            dict(
                ok=True,
                release=manifest["release"],
                upgrade_files=len(manifest["sha256"]),
                verified_base_files=len(manifest["base_sha256"]),
                protected_files=len(manifest["protected_sha256"]),
                algorithm_structure="MO-CAPO core with explicit fairness-readiness initialization adaptation",
                numerical_identity_to_two_objective_MOCAPO=False,
                gpu_performance_validated=False,
            ),
            indent=2,
        )
    )
    return True


if __name__ == "__main__":
    verify()
