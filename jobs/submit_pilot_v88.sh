#!/usr/bin/env bash
# Submit one model across seed x dataset tasks, validation, then aggregate reporting.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
model="${MODEL:-qwen-3-30b}"
budget="${PILOT_BUDGET:-2000000}"
concurrency="${MAX_CONCURRENT:-5}"
export PILOT_SEEDS="${PILOT_SEEDS:-${SEED:-72,73,74}}"
export PILOT_RECLAIM_RESERVE=1
source jobs/lib/pilot_v86_matrix.sh
tf_v86_matrix
seed="${tf_v86_seeds[0]}"
dry_run="${DRY_RUN:-1}"
[[ "$model" =~ ^(qwen-3-30b|mistral-3-24b|gpt-oss-120b)$ ]] || { echo "Invalid MODEL" >&2; exit 2; }
[[ "$budget" =~ ^[1-9][0-9]*$ && "$seed" =~ ^[0-9]+$ ]] || { echo "Invalid budget/seed" >&2; exit 2; }
[[ "$concurrency" =~ ^[1-9][0-9]{0,3}$ && "$dry_run" =~ ^[01]$ ]] && ((concurrency <= tf_v86_tasks)) || { echo "MAX_CONCURRENT must be 1-$tf_v86_tasks; DRY_RUN must be 0 or 1" >&2; exit 2; }
export MODEL="$model" PILOT_BUDGET="$budget" SEED="$seed"
export PILOT_SEEDS="$(IFS=,; echo "${tf_v86_seeds[*]}")"
export EVALUATION_SPLIT="${EVALUATION_SPLIT:-validation}"
[[ "$EVALUATION_SPLIT" == validation || "$EVALUATION_SPLIT" == test ]] || { echo "Invalid EVALUATION_SPLIT" >&2; exit 2; }
export PILOT_METHODS="${PILOT_METHODS:-Tri-Fair-v8.8,NSGAII-PO-Fair,NSGAIII-PO-Fair,SPEA2-PO-Fair,RandomSearch-PO-Fair}"
# Reject stale method exports before allocating any GPUs.
IFS=',' read -r -a selected_methods <<< "$PILOT_METHODS"
declare -A selected_seen=()
for selected_method in "${selected_methods[@]}"; do
    case "$selected_method" in
        Tri-Fair-v8.8|NSGAII-PO-Fair|NSGAIII-PO-Fair|SPEA2-PO-Fair|RandomSearch-PO-Fair) ;;
        *) echo "Invalid or historical v8.8 pilot method: $selected_method" >&2; exit 2 ;;
    esac
    [[ -z "${selected_seen[$selected_method]:-}" ]] || { echo "Duplicate method: $selected_method" >&2; exit 2; }
    selected_seen[$selected_method]=1
done
[[ "${selected_seen[Tri-Fair-v8.8]:-}" == 1 && ${#selected_methods[@]} -ge 2 ]] || { echo "Include Tri-Fair-v8.8 and at least one registered control" >&2; exit 2; }
# Distinct batch directories avoid ambiguous pilot selection and stale caches.
export PILOT_RESULTS_ROOT="${PILOT_RESULTS_ROOT:-results/tri_fair_v88_pilot_${budget}/batch_$(date -u +%Y%m%dT%H%M%S)_${RANDOM}}"
if [[ "$PILOT_METHODS" == *Hybrid* || "$PILOT_METHODS" == *Tri-Fair-v8.3* || "$PILOT_METHODS" == *Tri-Fair-v8.5* ]]; then
    echo "Use only the active v8.8 proposed method and comparison methods for this pilot." >&2
    exit 2
fi
args=(--parsable --export=ALL --array="0-$((tf_v86_tasks-1))%$concurrency")
if [[ -n "${PILOT_GPU_TYPE:-}" ]]; then
    [[ "$PILOT_GPU_TYPE" =~ ^(a100-80g|h200-141g|b200-180g)$ ]] || { echo "Invalid PILOT_GPU_TYPE" >&2; exit 2; }
    [[ "$model" != gpt-oss-120b || "$PILOT_GPU_TYPE" != a100-80g ]] || { echo "GPT-OSS requires a validated H200/B200 backend" >&2; exit 2; }
    args+=(--gpus-per-task="${PILOT_GPU_TYPE}:1")
fi
if [[ "$model" == "gpt-oss-120b" ]]; then
    # A100-80G is not validated for the 120B model. Request a large-memory
    # H200/B200 node after its backend smoke succeeds.
    args+=(--exclude='falcon[1-6],pegasus,pegasus2,shepard3')
fi
echo "Model=$model seeds=$PILOT_SEEDS per-method-cap=$budget methods=$PILOT_METHODS"
echo "Results=$PILOT_RESULTS_ROOT"
echo "Optimizer runs=$((tf_v86_tasks * ${#selected_methods[@]})); downstream ceiling=$((tf_v86_tasks * ${#selected_methods[@]} * budget)) tokens, plus meta and validation."
echo "$tf_v86_tasks seed/dataset tasks; $concurrency GPUs maximum per array; methods run sequentially inside each task."
if [[ "$dry_run" == 1 ]]; then
    printf 'DRY RUN: '; printf '%q ' sbatch "${args[@]}" jobs/pilot_v88.sbatch; printf '\n'
    printf 'DRY RUN: '; printf '%q ' sbatch "${args[@]}" --dependency=afterok:TRAIN_JOB jobs/pilot_v88_evaluate.sbatch; printf '\n'
    printf 'DRY RUN: '; printf '%q ' sbatch --parsable --export=ALL --dependency=afterok:VALIDATION_JOB jobs/pilot_v88_report.sbatch; printf '\n'
    exit 0
fi
python -m tri_fair.cli.preflight_v88
mkdir -p logs
train=$(sbatch "${args[@]}" jobs/pilot_v88.sbatch)
train=${train%%;*}
[[ "$train" =~ ^[0-9]+$ ]] || { echo "Unexpected training job ID: $train" >&2; exit 1; }
echo "Training array=$train"
if ! validation=$(sbatch "${args[@]}" --dependency="afterok:$train" jobs/pilot_v88_evaluate.sbatch); then
    echo "Training array $train was submitted; validation submission failed. Do not resubmit training." >&2
    exit 1
fi
validation=${validation%%;*}
echo "Validation array=$validation (waits for all $tf_v86_tasks training tasks)"
printf 'Inspect: sacct -X --array -j %s,%s --format=JobID,State,Elapsed,ExitCode\n' "$train" "$validation"
[[ "$validation" =~ ^[0-9]+$ ]] || { echo "Unexpected validation job ID: $validation" >&2; exit 1; }
if ! report=$(sbatch --parsable --export=ALL --dependency="afterok:$validation" jobs/pilot_v88_report.sbatch); then
    echo "Training $train and validation $validation were submitted; report submission failed. Do not resubmit training." >&2
    exit 1
fi
report=${report%%;*}
[[ "$report" =~ ^[0-9]+$ ]] || { echo "Unexpected report job ID: $report" >&2; exit 1; }
echo "Aggregate report=$report (waits for all validation tasks)"
printf 'Inspect all jobs: sacct -X --array -j %s,%s,%s --format=JobID,State,Elapsed,ExitCode\n' "$train" "$validation" "$report"

