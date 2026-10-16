"""Verify every file listed in the complete release checksum manifest."""

from __future__ import annotations
import argparse
import hashlib
from pathlib import Path, PurePosixPath


def verify(root):
    root = Path(root).resolve()
    checksum_file = root / "RELEASE_SHA256SUMS.txt"
    failures = []
    count = 0
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        relative_path = PurePosixPath(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("unsafe checksum path")
        path = root.joinpath(*relative_path.parts)
        if not path.is_file():
            failures.append(f"missing: {relative}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(f"changed: {relative}")
        count += 1
    if failures:
        raise RuntimeError("\n".join(failures))
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    print(f"Verified {verify(args.root)} release files.")


if __name__ == "__main__":
    main()
