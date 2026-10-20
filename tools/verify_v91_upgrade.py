"""Verify the installed v9.1 upgrade without loading a model or downloading data."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def verify(root=ROOT):
    root = Path(root)
    manifest = json.loads((root / 'UPGRADE_V91_INTENSIFICATION.json').read_text(encoding='utf-8'))
    failures = []
    for name, expected in manifest['sha256'].items():
        path = root / name
        if not path.is_file():
            failures.append(name + ' (missing)')
        elif hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() != expected:
            failures.append(name + ' (changed)')
    for name, expected in manifest['protected_sha256'].items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() != expected:
            failures.append(name + ' (protected comparator/metric changed)')
    if failures:
        raise SystemExit('Verification failed:\n' + '\n'.join(failures))
    print(json.dumps(dict(ok=True, release=manifest['release'],
                         upgrade_files=len(manifest['sha256']),
                         protected_files=len(manifest['protected_sha256']),
                         gpu_performance_validated=False), indent=2))


if __name__ == '__main__':
    verify()
