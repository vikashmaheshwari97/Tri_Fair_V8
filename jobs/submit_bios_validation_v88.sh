#!/usr/bin/env bash
# Retry ONLY held-out validation and figures for an already completed Bios batch.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
dry="${DRY_RUN:-1}"
[[ "$dry" =~ ^[01]$ ]] || { echo 'DRY_RUN must be 0 or 1' >&2; exit 2; }
export BIOS_OUTPUT="${BIOS_OUTPUT:-results/tri_fair_v88_stable_bios_2000000/batch_20261020T071902_3371}"
export STABILITY_STAGE=evaluate VLLM_BATCH_INVARIANT=1
# Discover the exact proof from the completed training batch, never the newest glob.
PROBE_PROOF=$(python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["BIOS_OUTPUT"]).expanduser().resolve()
def read(path):
    return json.loads(path.read_text(encoding="utf-8"))
profile = read(root / "stable_profile.json")
ready = read(root / "ready_for_validation.json")
if profile.get("batch_invariant") is not True or profile.get("dataset") != "bias_in_bios" or profile.get("seed") != 72 or profile.get("budget") != 2000000:
    raise SystemExit("Saved training profile differs from the matched Bios recovery")
report_path = Path(ready["report"]).resolve()
if not report_path.is_relative_to(root):
    raise SystemExit("Training report is outside the selected batch")
report = read(report_path)
expected = {"Tri-Fair-v8.8", "NSGAII-PO-Fair", "NSGAIII-PO-Fair", "SPEA2-PO-Fair", "RandomSearch-PO-Fair"}
rows = report.get("runs", [])
if not report.get("all_controls_comparable") or len(rows) != 5 or {r.get("method") for r in rows} != expected:
    raise SystemExit("The completed report must contain exactly five comparable methods")
if any(r.get("status") != "measured" or r.get("process_exit_code") != 0 for r in rows):
    raise SystemExit("Training is not complete; validation-only submission refused")
proof = Path(profile["proof_path"]).expanduser().resolve()
import hashlib
digest = hashlib.sha256(proof.read_bytes()).hexdigest()
if ready.get("proof_sha256") != digest or profile.get("proof_sha256") != digest:
    raise SystemExit("Saved training and validation proof hashes differ")
print(proof)
PY
)
export PROBE_PROOF
# Preserves the existing checks for source, packages, manifest and launcher.
python recovery/v88_bios_stability.py validate --proof "$PROBE_PROOF"
echo "Validation only: $BIOS_OUTPUT"
echo 'Training runs reused: 5; optimizer runs submitted: 0; stability probe reruns: 0.'
command=(sbatch --parsable --export=ALL --job-name=tfv88-valid jobs/bios_stability_v88.sbatch)
if [[ "$dry" == 1 ]]; then
  printf 'DRY RUN: '; printf '%q ' "${command[@]}"; printf '\n'; exit 0
fi
mkdir -p logs
job=$("${command[@]}")
job=${job%%;*}
[[ "$job" =~ ^[0-9]+$ ]] || { echo "Unexpected validation job ID: $job" >&2; exit 1; }
echo "Validation and figure job=$job"
printf 'Inspect: sacct -X -j %s --format=JobID,JobName,State,Elapsed,ExitCode\n' "$job"
