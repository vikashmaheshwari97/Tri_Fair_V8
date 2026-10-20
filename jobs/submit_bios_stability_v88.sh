#!/usr/bin/env bash
# Default: submit only a small diagnostic, never an optimizer matrix.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode="${MODE:-probe}"
dry="${DRY_RUN:-1}"
[[ "$dry" =~ ^[01]$ ]] || { echo 'DRY_RUN must be 0 or 1' >&2; exit 2; }
[[ "$mode" == probe || "$mode" == train ]] || { echo 'MODE must be probe or train' >&2; exit 2; }
export MANIFEST_DIR="${MANIFEST_DIR:-data/splits_v8_clean}"
export VLLM_BATCH_INVARIANT=1
stamp="$(date -u +%Y%m%dT%H%M%S)_${RANDOM}"
mkdir -p logs
if [[ "$mode" == probe ]]; then
  export ORIGINAL_PROBE="${ORIGINAL_PROBE:-results/tri_fair_v88_pilot_2000000/batch_20261019T223036_12131/qwen-3-30b/bias_in_bios/seed72/20261019T225325Z_081b0486/Tri-Fair-v8.8/run/repeatability_probe.json}"
  [[ -s "$ORIGINAL_PROBE" ]] || { echo "Missing original probe: $ORIGINAL_PROBE" >&2; exit 2; }
  export PROBE_OUTPUT="${PROBE_OUTPUT:-results/tri_fair_v88_stability/probe_$stamp}"
  [[ ! -e "$PROBE_OUTPUT" ]] || { echo 'Use a fresh PROBE_OUTPUT' >&2; exit 2; }
  export STABILITY_STAGE=probe
  echo "Probe only: four initial prompts, two rounds, 192 predictions maximum."
  echo "Proof after success: $PROBE_OUTPUT/stability_proof.json"
  command=(sbatch --parsable --export=ALL --time=00:30:00 --job-name=tfv88-probe jobs/bios_stability_v88.sbatch)
  if [[ "$dry" == 1 ]]; then
    printf 'DRY RUN: '; printf '%q ' "${command[@]}"; printf '\n'; exit 0
  fi
  job=$("${command[@]}")
  echo "Probe job=${job%%;*}"
  echo "No training or validation jobs were submitted."
  exit 0
fi
: "${PROBE_PROOF:?Set PROBE_PROOF to the passing stability_proof.json}"
export PROBE_PROOF
# Cheap checks on the login node. No torch, model or CUDA imports.
python recovery/v88_bios_stability.py validate --proof "$PROBE_PROOF"
export BIOS_OUTPUT="${BIOS_OUTPUT:-results/tri_fair_v88_stable_bios_2000000/batch_$stamp}"
[[ ! -e "$BIOS_OUTPUT" ]] || { echo 'Use a fresh BIOS_OUTPUT' >&2; exit 2; }
export STABILITY_STAGE=train
echo "Five matched Bios methods; seed72; 2M cap each; H200; batch invariance ON."
echo "New results: $BIOS_OUTPUT"
echo 'Old results remain separate. No old baseline results will be merged.'
if [[ "$dry" == 1 ]]; then
  echo 'DRY RUN: sbatch --parsable --export=ALL --job-name=tfv88-bios jobs/bios_stability_v88.sbatch'
  echo 'DRY RUN: sbatch --parsable --export=ALL --dependency=afterok:TRAIN_JOB --kill-on-invalid-dep=yes --job-name=tfv88-valid jobs/bios_stability_v88.sbatch'
  exit 0
fi
train=$(sbatch --parsable --export=ALL --job-name=tfv88-bios jobs/bios_stability_v88.sbatch)
train=${train%%;*}
[[ "$train" =~ ^[0-9]+$ ]] || { echo "Unexpected job ID: $train" >&2; exit 1; }
echo "Training job=$train"
export STABILITY_STAGE=evaluate
if ! validation=$(sbatch --parsable --export=ALL --dependency="afterok:$train" --kill-on-invalid-dep=yes --job-name=tfv88-valid jobs/bios_stability_v88.sbatch); then
  echo "Training $train was submitted; validation submission failed. Do not resubmit training." >&2; exit 1
fi
validation=${validation%%;*}
echo "Validation and figure job=$validation"
printf 'Inspect: sacct -X -j %s,%s --format=JobID,State,Elapsed,ExitCode\n' "$train" "$validation"
