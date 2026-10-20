#!/usr/bin/env bash
# Recover one probe-failed method, then evaluate the saved twenty-run matrix.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${RECOVERY_BATCH:?Set RECOVERY_BATCH to the original v8.8 batch directory}"
dry="${DRY_RUN:-1}"
[[ "$dry" =~ ^[01]$ ]] || { echo 'DRY_RUN must be 0 or 1' >&2; exit 2; }
export RECOVERY_BATCH
export MODEL=qwen-3-30b PILOT_SEEDS=72 SEED=72 PILOT_BUDGET=2000000
export EVALUATION_SPLIT=validation
export PILOT_METHODS=Tri-Fair-v8.8,NSGAII-PO-Fair,NSGAIII-PO-Fair,SPEA2-PO-Fair,RandomSearch-PO-Fair
unset PILOT_STAMP
export PILOT_RESULTS_ROOT="${RECOVERY_OUTPUT_ROOT:-$RECOVERY_BATCH/recovery_$(date -u +%Y%m%dT%H%M%S)_${RANDOM}}"
python recovery/v88_recover.py plan --batch-root "$RECOVERY_BATCH" --output-root "$PILOT_RESULTS_ROOT"
family="${RECOVERY_GPU_TYPE:-h200-141g}"
[[ "$family" =~ ^(h200-141g|b200-180g|a100-80g|auto)$ ]] || { echo 'Invalid RECOVERY_GPU_TYPE' >&2; exit 2; }
gpu_args=(--parsable --export=ALL)
[[ "$family" == auto ]] || gpu_args+=(--gpus-per-task="$family:1")
echo "Retry: only Tri-Fair-v8.8 / bias_in_bios / Qwen / seed72 / 2M cap"
echo "Original results: $RECOVERY_BATCH"
echo "Recovery results: $PILOT_RESULTS_ROOT"
echo "GPU family: $family; repeatability gate remains mandatory"
if [[ "$dry" == 1 ]]; then
    printf 'DRY RUN: '; printf '%q ' sbatch "${gpu_args[@]}" jobs/recover_v88.sbatch; printf '\n'
    printf 'DRY RUN: '; printf '%q ' sbatch --parsable --export=ALL --array=0-3%2 --dependency=afterok:RETRY_JOB jobs/pilot_v88_evaluate.sbatch; printf '\n'
    printf 'DRY RUN: '; printf '%q ' sbatch --parsable --export=ALL --dependency=afterok:VALIDATION_JOB jobs/pilot_v88_report.sbatch; printf '\n'
    exit 0
fi
mkdir -p logs
retry=$(sbatch "${gpu_args[@]}" jobs/recover_v88.sbatch)
retry=${retry%%;*}
[[ "$retry" =~ ^[0-9]+$ ]] || { echo "Unexpected retry ID: $retry" >&2; exit 1; }
echo "Recovery job=$retry"
if ! validation=$(sbatch --parsable --export=ALL --array=0-3%2 --dependency="afterok:$retry" jobs/pilot_v88_evaluate.sbatch); then
    echo "Retry $retry submitted; validation submission failed. Do not resubmit the optimizer." >&2; exit 1
fi
validation=${validation%%;*}
[[ "$validation" =~ ^[0-9]+$ ]] || { echo "Unexpected validation ID: $validation" >&2; exit 1; }
echo "Validation array=$validation"
if ! report=$(sbatch --parsable --export=ALL --dependency="afterok:$validation" jobs/pilot_v88_report.sbatch); then
    echo "Retry $retry and validation $validation submitted; report submission failed." >&2; exit 1
fi
report=${report%%;*}
[[ "$report" =~ ^[0-9]+$ ]] || { echo "Unexpected report ID: $report" >&2; exit 1; }
echo "Report job=$report"
printf 'Inspect: sacct -X --array -j %s,%s,%s --format=JobID,State,Elapsed,ExitCode\n' "$retry" "$validation" "$report"
