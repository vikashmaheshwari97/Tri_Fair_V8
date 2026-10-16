#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source jobs/lib/common.sh

export TF_MODELS="${TF_MODELS:-qwen-3-30b,mistral-3-24b,gpt-oss-120b}"
export TF_DATASETS="${TF_DATASETS:-bbq,civil_comments,bias_in_bios}"
export TF_OPTIMIZERS="${TF_OPTIMIZERS:-Tri-Fair-v8,NSGAII-PO-Fair,SPEA2-PO-Fair,NSGAIII-PO-Fair,RandomSearch-PO-Fair,NSGAII-PO-Fair-Enhanced}"
export TF_SEEDS="${TF_SEEDS:-72,73,74}"
export BUDGET="${BUDGET:-5000000}"
export RUN_MODE="${RUN_MODE:-fresh}"
export MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
export MANIFEST_DIR="${MANIFEST_DIR:-data/splits_v8_clean}"
export MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-16}"
export META_MAX_OUTPUT_TOKENS="${META_MAX_OUTPUT_TOKENS:-256}"
export N_INIT_PROMPTS="${N_INIT_PROMPTS:-12}"
export TF_RESULTS_NAMESPACE="${TF_RESULTS_NAMESPACE:-tri_fair_v8_clean_5m}"
export MAX_STEPS="${MAX_STEPS:-4000}"
export STRICT_TOKEN_BUDGET="${STRICT_TOKEN_BUDGET:-1}"
export NEAR_BUDGET_FRACTION="${NEAR_BUDGET_FRACTION:-0.90}"
export MIN_BUDGET_UTILIZATION="${MIN_BUDGET_UTILIZATION:-0.90}"
export OUTPUT_TOKEN_RESERVE="${OUTPUT_TOKEN_RESERVE:-2}"
export FORCE_FRESH="${FORCE_FRESH:-0}"
export DRY_RUN="${DRY_RUN:-1}"
export NODELIST="${NODELIST:-}"
export CPUS_PER_TASK="${CPUS_PER_TASK:-16}"

declare -a models datasets optimizers seeds
tf_split_csv "$TF_MODELS" models
tf_split_csv "$TF_DATASETS" datasets
tf_split_csv "$TF_OPTIMIZERS" optimizers
tf_split_csv "$TF_SEEDS" seeds
total=$((${#models[@]} * ${#datasets[@]} * ${#optimizers[@]} * ${#seeds[@]}))
array_spec="0-$((total - 1))%$MAX_CONCURRENT"
mkdir -p logs "results/$TF_RESULTS_NAMESPACE/submissions"

SBATCH_ARGS=("--gpus-per-task=1" --parsable "--array=$array_spec" "--job-name=tfv8-${BUDGET}")
[[ -n "${PARTITION:-}" ]] && SBATCH_ARGS+=("--partition=$PARTITION")
[[ -n "${TIME_LIMIT:-}" ]] && SBATCH_ARGS+=("--time=$TIME_LIMIT")
[[ -n "${MEMORY:-}" ]] && SBATCH_ARGS+=("--mem=$MEMORY")
[[ -n "${CPUS_PER_TASK:-}" ]] && SBATCH_ARGS+=("--cpus-per-task=$CPUS_PER_TASK")
[[ -n "${NODELIST:-}" ]] && SBATCH_ARGS+=("--nodelist=$NODELIST")

printf 'Tri-Fair-v8 matched submission\n'
printf '  models: %s\n  datasets: %s\n  optimizers: %s\n' \
  "$TF_MODELS" "$TF_DATASETS" "$TF_OPTIMIZERS"
printf '  seeds: %s\n  budget: %s\n  namespace: %s\n' \
  "$TF_SEEDS" "$BUDGET" "$TF_RESULTS_NAMESPACE"
printf '  array: %s (%d tasks)\n' "$array_spec" "$total"

if tf_is_true "$DRY_RUN"; then
  printf 'DRY RUN: sbatch'
  printf ' %q' "${SBATCH_ARGS[@]}" jobs/train.sbatch
  printf '\n'
  exit 0
fi

job_id="$(sbatch "${SBATCH_ARGS[@]}" jobs/train.sbatch)"
tf_log "Submitted Tri-Fair-v8 array job $job_id"
