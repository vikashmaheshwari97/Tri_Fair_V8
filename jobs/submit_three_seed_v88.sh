#!/usr/bin/env bash
# Frozen Qwen matrix: 12 seed/dataset tasks, five matched methods per task.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export MODEL="${MODEL:-qwen-3-30b}"
export PILOT_SEEDS="${PILOT_SEEDS:-72,73,74}"
export PILOT_BUDGET="${PILOT_BUDGET:-2000000}"
export MAX_CONCURRENT="${MAX_CONCURRENT:-5}"
export MANIFEST_DIR="${MANIFEST_DIR:-data/splits_v8_clean}"
export VLLM_BATCH_INVARIANT=1
export MPLBACKEND=Agg
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
dry="${DRY_RUN:-1}"
[[ "$dry" =~ ^[01]$ ]] || { echo 'DRY_RUN must be 0 or 1' >&2; exit 2; }
# --exclude means ANY eligible node, unlike --nodelist which requests all listed nodes.
args=(--parsable --export=ALL --array="0-11%$MAX_CONCURRENT" --exclude='falcon[1-6],pegasus,shepard3')
if [[ -n "${PILOT_GPU_TYPE:-}" ]]; then
  [[ "$PILOT_GPU_TYPE" =~ ^(a100-80g|h200-141g|b200-180g)$ ]] || { echo 'Invalid PILOT_GPU_TYPE' >&2; exit 2; }
  args+=(--gpus-per-task="${PILOT_GPU_TYPE}:1")
fi
python research/run_three_seed_v88.py plan
if [[ "$dry" == 1 ]]; then
  printf 'DRY RUN: '; printf '%q ' sbatch "${args[@]}" jobs/three_seed_v88.sbatch; printf '\n'
  printf 'DRY RUN: '; printf '%q ' sbatch "${args[@]}" --dependency=afterok:TRAIN_JOB --kill-on-invalid-dep=yes --job-name=tfv88-valid jobs/three_seed_v88.sbatch; printf '\n'
  printf 'DRY RUN: '; printf '%q ' sbatch --parsable --export=ALL --dependency=afterok:VALIDATION_JOB --kill-on-invalid-dep=yes jobs/three_seed_v88_report.sbatch; printf '\n'
  exit 0
fi
python -m tri_fair.cli.preflight_v88
export PILOT_RESULTS_ROOT="${PILOT_RESULTS_ROOT:-results/tri_fair_v88_three_seed_2000000/batch_$(date -u +%Y%m%dT%H%M%S)_$RANDOM}"
python research/run_three_seed_v88.py init
mkdir -p logs
export TF_MATRIX_STAGE=train
train=$(sbatch "${args[@]}" jobs/three_seed_v88.sbatch); train=${train%%;*}
[[ "$train" =~ ^[0-9]+$ ]] || { echo "Invalid training job: $train" >&2; exit 1; }
echo "Training array=$train"
export TF_MATRIX_STAGE=evaluate
if ! valid=$(sbatch "${args[@]}" --dependency="afterok:$train" --kill-on-invalid-dep=yes --job-name=tfv88-valid jobs/three_seed_v88.sbatch); then
  echo "Training $train already submitted. Retry validation only; do not repeat training." >&2; exit 1
fi
valid=${valid%%;*}; [[ "$valid" =~ ^[0-9]+$ ]] || exit 1
if ! report=$(sbatch --parsable --export=ALL --dependency="afterok:$valid" --kill-on-invalid-dep=yes jobs/three_seed_v88_report.sbatch); then
  echo "Training $train and validation $valid already submitted. Retry report only." >&2; exit 1
fi
report=${report%%;*}; [[ "$report" =~ ^[0-9]+$ ]] || exit 1
python research/run_three_seed_v88.py record --job-ids "$train,$valid,$report"
echo "Validation array=$valid; aggregate report=$report"
echo "Results=$PILOT_RESULTS_ROOT"
printf 'Inspect: sacct -X --array -j %s,%s,%s --format=JobID,State,Elapsed,ExitCode\n' "$train" "$valid" "$report"
