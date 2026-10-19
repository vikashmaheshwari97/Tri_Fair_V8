#!/usr/bin/env bash
# Submit one model, four dataset tasks, and a dependent validation array.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
model="${MODEL:-qwen-3-30b}"
budget="${PILOT_BUDGET:-2000000}"
concurrency="${MAX_CONCURRENT:-2}"
seed="${SEED:-72}"
dry_run="${DRY_RUN:-1}"
[[ "$model" =~ ^(qwen-3-30b|mistral-3-24b|gpt-oss-120b)$ ]] || { echo "Invalid MODEL" >&2; exit 2; }
[[ "$budget" =~ ^[1-9][0-9]*$ && "$seed" =~ ^[0-9]+$ ]] || { echo "Invalid budget/seed" >&2; exit 2; }
[[ "$concurrency" =~ ^[1-4]$ && "$dry_run" =~ ^[01]$ ]] || { echo "MAX_CONCURRENT must be 1-4; DRY_RUN must be 0 or 1" >&2; exit 2; }
export MODEL="$model" PILOT_BUDGET="$budget" SEED="$seed"
export PILOT_METHODS="${PILOT_METHODS:-Tri-Fair-v8.6,NSGAII-PO-Fair}"
# Distinct batch directories avoid ambiguous pilot selection and stale caches.
export PILOT_RESULTS_ROOT="${PILOT_RESULTS_ROOT:-results/tri_fair_v86_pilot_${budget}/batch_$(date -u +%Y%m%dT%H%M%S)_${RANDOM}}"
if [[ "$PILOT_METHODS" == *Hybrid* || "$PILOT_METHODS" == *Tri-Fair-v8.3* || "$PILOT_METHODS" == *Tri-Fair-v8.5* ]]; then
    echo "Hybrid is retired from v8.5 pilots. Use the historical v8.4 entrypoint for reproduction." >&2
    exit 2
fi
args=(--parsable --export=ALL --array="0-3%$concurrency")
if [[ "$model" == "gpt-oss-120b" ]]; then
    # A100-80G is not validated for the 120B model. Request a large-memory
    # H200/B200 node after its backend smoke succeeds.
    args+=(--exclude='falcon[1-6],pegasus,pegasus2,shepard3')
fi
echo "Model=$model seed=$seed per-method-cap=$budget methods=$PILOT_METHODS"
echo "Results=$PILOT_RESULTS_ROOT"
echo "Four dataset tasks; $concurrency GPUs maximum; methods run sequentially inside each task."
if [[ "$dry_run" == 1 ]]; then
    printf 'DRY RUN: '; printf '%q ' sbatch "${args[@]}" jobs/pilot_v86.sbatch; printf '\n'
    printf 'DRY RUN: '; printf '%q ' sbatch "${args[@]}" --dependency=afterok:TRAIN_JOB jobs/pilot_v86_evaluate.sbatch; printf '\n'
    exit 0
fi
mkdir -p logs
train=$(sbatch "${args[@]}" jobs/pilot_v86.sbatch)
train=${train%%;*}
[[ "$train" =~ ^[0-9]+$ ]] || { echo "Unexpected training job ID: $train" >&2; exit 1; }
echo "Training array=$train"
if ! validation=$(sbatch "${args[@]}" --dependency="afterok:$train" jobs/pilot_v86_evaluate.sbatch); then
    echo "Training array $train was submitted; validation submission failed. Do not resubmit training." >&2
    exit 1
fi
validation=${validation%%;*}
echo "Validation array=$validation (waits for all four training tasks)"
printf 'Inspect: sacct -X --array -j %s,%s --format=JobID,State,Elapsed,ExitCode\n' "$train" "$validation"
