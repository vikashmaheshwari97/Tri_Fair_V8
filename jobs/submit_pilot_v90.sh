#!/usr/bin/env bash
# Fresh matched screening: default seed72, all four datasets and five methods.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
for setting in NODELIST GPU_TYPE SBATCH_NODELIST SBATCH_GPUS_PER_TASK; do
  if [[ -n "${!setting:-}" ]]; then
    echo "Unset $setting before this matched launch; use PILOT_GPU_TYPE only for an explicit GPU-family restriction." >&2
    exit 2
  fi
done
export V90_PROFILE="${V90_PROFILE:-ablation}"
export PILOT_GPU_TYPE="${PILOT_GPU_TYPE:-h200-141g}"
export MODEL="${MODEL:-qwen-3-30b}"
export PILOT_SEEDS="${PILOT_SEEDS:-72}"
export PILOT_BUDGET="${PILOT_BUDGET:-2000000}"
export MAX_CONCURRENT="${MAX_CONCURRENT:-5}"
export MANIFEST_DIR="${MANIFEST_DIR:-data/splits_v8_clean}"
export VLLM_BATCH_INVARIANT=1
export MPLBACKEND=Agg
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
dry="${DRY_RUN:-1}"
[[ "$dry" =~ ^[01]$ ]] || { echo 'DRY_RUN must be 0 or 1' >&2; exit 2; }
# --exclude means ANY eligible node, unlike --nodelist which requests all listed nodes.
IFS=, read -r -a seed_values <<< "$PILOT_SEEDS"
last=$((4 * ${#seed_values[@]} - 1))
args=(--parsable --export=ALL --array="0-$last%$MAX_CONCURRENT" --exclude='falcon[1-6],pegasus,shepard3')
if [[ -n "${PILOT_GPU_TYPE:-}" ]]; then
  [[ "$PILOT_GPU_TYPE" =~ ^(a100-80g|h200-141g|b200-180g)$ ]] || { echo 'Invalid PILOT_GPU_TYPE' >&2; exit 2; }
  args+=(--gpus-per-task="${PILOT_GPU_TYPE}:1")
fi
python research/run_pilot_v90.py plan
if [[ "$dry" == 1 ]]; then
  printf 'DRY RUN: '; printf '%q ' sbatch "${args[@]}" jobs/pilot_v90.sbatch; printf '\n'
  printf 'DRY RUN: '; printf '%q ' sbatch "${args[@]}" --dependency=afterok:TRAIN_JOB --kill-on-invalid-dep=yes --job-name=tfv90-valid jobs/pilot_v90.sbatch; printf '\n'
  printf 'DRY RUN: '; printf '%q ' sbatch --parsable --export=ALL --dependency=afterok:VALIDATION_JOB --kill-on-invalid-dep=yes jobs/pilot_v90_report.sbatch; printf '\n'
  exit 0
fi
python -m tri_fair.cli.preflight_v88
python -m tri_fair.cli.preflight_v88
export PILOT_RESULTS_ROOT="${PILOT_RESULTS_ROOT:-results/tri_fair_v90_${V90_PROFILE}_${MODEL}_2000000/batch_$(date -u +%Y%m%dT%H%M%S)_$RANDOM}"
python research/run_pilot_v90.py init
mkdir -p logs
export TF_MATRIX_STAGE=train
train=$(sbatch "${args[@]}" jobs/pilot_v90.sbatch); train=${train%%;*}
[[ "$train" =~ ^[0-9]+$ ]] || { echo "Invalid training job: $train" >&2; exit 1; }
echo "Training array=$train"
export TF_MATRIX_STAGE=evaluate
if ! valid=$(sbatch "${args[@]}" --dependency="afterok:$train" --kill-on-invalid-dep=yes --job-name=tfv90-valid jobs/pilot_v90.sbatch); then
  echo "Training $train already submitted. Retry validation only; do not repeat training." >&2; exit 1
fi
valid=${valid%%;*}; [[ "$valid" =~ ^[0-9]+$ ]] || exit 1
if ! report=$(sbatch --parsable --export=ALL --dependency="afterok:$valid" --kill-on-invalid-dep=yes jobs/pilot_v90_report.sbatch); then
  echo "Training $train and validation $valid already submitted. Retry report only." >&2; exit 1
fi
report=${report%%;*}; [[ "$report" =~ ^[0-9]+$ ]] || exit 1
python research/run_pilot_v90.py record --job-ids "$train,$valid,$report"
echo "Validation array=$valid; aggregate report=$report"
echo "Results=$PILOT_RESULTS_ROOT"
printf 'Inspect: sacct -X --array -j %s,%s,%s --format=JobID,State,Elapsed,ExitCode\n' "$train" "$valid" "$report"
