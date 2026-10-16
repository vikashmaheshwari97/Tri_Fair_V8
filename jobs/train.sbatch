#!/usr/bin/env bash
#SBATCH --job-name=tri-fair-v8
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=logs/tri_fair_v8_%A_%a.out
#SBATCH --error=logs/tri_fair_v8_%A_%a.err

set -Eeuo pipefail
module purge
module --ignore_cache load python/3.11.14
module --ignore_cache load python-venv/1.0
source "${TF_VENV:-$HOME/venvs/tri-fair-v8}/bin/activate"
export TF_PYTHON="${TF_VENV:-$HOME/venvs/tri-fair-v8}/bin/python"

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
source jobs/lib/common.sh
source jobs/lib/model_cache.sh

TF_MODELS="${TF_MODELS:-qwen-3-30b,mistral-3-24b}"
TF_DATASETS="${TF_DATASETS:-bbq,civil_comments,bias_in_bios}"
TF_OPTIMIZERS="${TF_OPTIMIZERS:-Tri-Fair-v8,NSGAII-PO-Fair,SPEA2-PO-Fair,NSGAIII-PO-Fair,RandomSearch-PO-Fair,NSGAII-PO-Fair-Enhanced}"
TF_SEEDS="${TF_SEEDS:-72,73,74}"
BUDGET="${BUDGET:-5000000}"
RUN_MODE="${RUN_MODE:-fresh}"
MANIFEST_DIR="${MANIFEST_DIR:-data/splits_v8_clean}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-16}"
META_MAX_OUTPUT_TOKENS="${META_MAX_OUTPUT_TOKENS:-256}"
N_INIT_PROMPTS="${N_INIT_PROMPTS:-12}"
TF_RESULTS_NAMESPACE="${TF_RESULTS_NAMESPACE:-tri_fair_v8_clean_5m}"
MAX_STEPS="${MAX_STEPS:-4000}"
FORCE_FRESH="${FORCE_FRESH:-0}"
STRICT_TOKEN_BUDGET="${STRICT_TOKEN_BUDGET:-1}"
NEAR_BUDGET_FRACTION="${NEAR_BUDGET_FRACTION:-0.90}"
MIN_BUDGET_UTILIZATION="${MIN_BUDGET_UTILIZATION:-0.90}"
OUTPUT_TOKEN_RESERVE="${OUTPUT_TOKEN_RESERVE:-2}"

declare -a models datasets optimizers seeds
tf_split_csv "$TF_MODELS" models
tf_split_csv "$TF_DATASETS" datasets
tf_split_csv "$TF_OPTIMIZERS" optimizers
tf_split_csv "$TF_SEEDS" seeds

n_models=${#models[@]}; n_datasets=${#datasets[@]}
n_optimizers=${#optimizers[@]}; n_seeds=${#seeds[@]}
total=$((n_models * n_datasets * n_optimizers * n_seeds))
task_id="${SLURM_ARRAY_TASK_ID:-0}"
((task_id >= 0 && task_id < total)) || tf_die "Array task $task_id outside matrix size $total"

idx=$task_id
seed_idx=$((idx % n_seeds)); idx=$((idx / n_seeds))
optimizer_idx=$((idx % n_optimizers)); idx=$((idx / n_optimizers))
dataset_idx=$((idx % n_datasets)); idx=$((idx / n_datasets))
model_idx=$((idx % n_models))

model="${models[$model_idx]}"
dataset="${datasets[$dataset_idx]}"
optimizer="${optimizers[$optimizer_idx]}"
seed="${seeds[$seed_idx]}"

tf_configure_model_cache "$model"
tf_contains "$optimizer" Tri-Fair-v8 NSGAII-PO-Fair SPEA2-PO-Fair NSGAIII-PO-Fair RandomSearch-PO-Fair NSGAII-PO-Fair-Enhanced Tri-Fair-v8-NoFeedback Tri-Fair-v8-NoDemo Tri-Fair-v8-NoUncertainty Tri-Fair-v8-FullEval \
  || tf_die "Unsupported v8 optimizer '$optimizer'"

checkpoints="$(tf_stage_checkpoints "$BUDGET")"
output_dir="$(tf_run_output_dir "$ROOT" "$model" "$dataset" "$optimizer" "$seed")"
status_dir="$output_dir/status"
mkdir -p "$output_dir" "$status_dir" logs

if command -v flock >/dev/null 2>&1; then
  exec 9>"$output_dir/.run.lock"
  flock -n 9 || tf_die "Another process holds $output_dir/.run.lock"
fi

logging_dir=""
latest_checkpoint=""
if logging_dir="$(tf_read_logging_dir "$output_dir" 2>/dev/null)"; then
  latest_checkpoint="$logging_dir/checkpoints/latest.pkl"
fi
resolved_mode="$RUN_MODE"
if [[ "$RUN_MODE" == "auto" ]]; then
  if [[ -n "$logging_dir" && -f "$latest_checkpoint" ]]; then
    resolved_mode=resume
  else
    resolved_mode=fresh
  fi
fi

resume_args=()
if [[ "$resolved_mode" == "resume" ]]; then
  [[ -f "$latest_checkpoint" ]] || tf_die "Missing checkpoint $latest_checkpoint"
  resume_args=(--resume-from "$latest_checkpoint")
elif [[ -n "$logging_dir" && -e "$logging_dir/step_results.parquet" ]] \
     && ! tf_is_true "$FORCE_FRESH"; then
  tf_die "Existing run found at $logging_dir; use resume or FORCE_FRESH=1"
fi

strict_budget_args=(--strict-token-budget)
if ! tf_is_true "$STRICT_TOKEN_BUDGET"; then
  strict_budget_args=(--no-strict-token-budget)
fi

declare -a PYTHON_CMD
tf_python_command PYTHON_CMD
"${PYTHON_CMD[@]}" -m tri_fair.cli.train \
  --experiment-name "tri_fair_v8_${optimizer}_${model}_${dataset}_${seed}_${BUDGET}" \
  --random-seed "$seed" \
  --budget-per-run "$BUDGET" \
  --budget-checkpoints "$checkpoints" \
  --output-dir "$output_dir" \
  --dataset "$dataset" \
  --model "$model" \
  --optimizer "$optimizer" \
  --n-init-prompts "$N_INIT_PROMPTS" \
  --max-output-tokens "$MAX_OUTPUT_TOKENS" \
  --meta-max-output-tokens "$META_MAX_OUTPUT_TOKENS" \
  --manifest-dir "$MANIFEST_DIR" \
  --max-steps "$MAX_STEPS" \
  --near-budget-fraction "$NEAR_BUDGET_FRACTION" \
  --minimum-budget-utilization "$MIN_BUDGET_UTILIZATION" \
  --output-token-reserve "$OUTPUT_TOKEN_RESERVE" \
  "${strict_budget_args[@]}" \
  "${resume_args[@]}"
