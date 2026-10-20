#!/usr/bin/env bash
# Retry evaluation/report only, using a saved frozen matrix. No training submitted.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${PILOT_RESULTS_ROOT:?Set this to the exact completed three-seed batch}"
export PILOT_RESULTS_ROOT VLLM_BATCH_INVARIANT=1 MPLBACKEND=Agg
export MODEL=qwen-3-30b
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
dry="${DRY_RUN:-1}"; stage="${STAGE:-evaluate}"
[[ "$dry" =~ ^[01]$ && "$stage" =~ ^(evaluate|report)$ ]] || exit 2
python research/run_three_seed_v88.py check
command=(sbatch --parsable --export=ALL)
if [[ "$stage" == evaluate ]]; then
  export TF_MATRIX_STAGE=evaluate
  command+=(--array=0-11%5 --exclude='falcon[1-6],pegasus,shepard3' --job-name=tfv88-valid jobs/three_seed_v88.sbatch)
else
  command+=(jobs/three_seed_v88_report.sbatch)
fi
if [[ "$dry" == 1 ]]; then
  printf 'DRY RUN: '; printf '%q ' "${command[@]}"; printf '\n'; exit 0
fi
mkdir -p logs
job=$("${command[@]}"); job=${job%%;*}
[[ "$job" =~ ^[0-9]+$ ]] || exit 1
echo "$stage job=$job; optimizer runs submitted=0"
if [[ "$stage" == evaluate ]]; then
  report=$(sbatch --parsable --export=ALL --dependency="afterok:$job" --kill-on-invalid-dep=yes jobs/three_seed_v88_report.sbatch)
  echo "Report job=${report%%;*}"
fi
