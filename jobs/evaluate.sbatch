#!/usr/bin/env bash
#SBATCH --job-name=tfv8-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --exclude=falcon[1-6],pegasus,shepard3
#SBATCH --hint=nomultithread
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=logs/tri_fair_v8_eval_%A_%a.out
#SBATCH --error=logs/tri_fair_v8_eval_%A_%a.err
set -Eeuo pipefail
module purge
module --ignore_cache load python/3.11.14
module --ignore_cache load python-venv/1.0
source "${TF_VENV:-$HOME/venvs/tri-fair-v8}/bin/activate"
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
source jobs/lib/common.sh
tf_require_research_gpu_node
tf_configure_cpp_runtime
source jobs/lib/model_cache.sh
tf_log "Allocation: job=${SLURM_JOB_ID:-local} nodes=${SLURM_JOB_NODELIST:-unknown}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -m tri_fair.cli.doctor --gpu

declare -a models datasets seeds
tf_split_csv "${TF_MODELS:-qwen-3-30b,mistral-3-24b,gpt-oss-120b}" models
tf_split_csv "${TF_DATASETS:-bbq,civil_comments,bias_in_bios}" datasets
tf_split_csv "${TF_SEEDS:-72,73,74}" seeds
idx="${SLURM_ARRAY_TASK_ID:-0}"
total=$((${#models[@]} * ${#datasets[@]} * ${#seeds[@]}))
(( idx >= 0 && idx < total )) || tf_die "Array task outside evaluation matrix"
seed="${seeds[$((idx % ${#seeds[@]}))]}"; idx=$((idx / ${#seeds[@]}))
dataset="${datasets[$((idx % ${#datasets[@]}))]}"; idx=$((idx / ${#datasets[@]}))
model="${models[$idx]}"
tf_configure_model_cache "$model"
python -m tri_fair.cli.evaluate \
  --study-root "${V8_RESULTS_ROOT:-results/tri_fair_v8_clean_5m}/$model" \
  --model "$model" --dataset "$dataset" --random-seed "$seed" \
  --manifest-dir "${MANIFEST_DIR:-data/splits_v8_clean}" \
  --optimizers "${TF_OPTIMIZERS:-Tri-Fair-v8,NSGAII-PO-Fair,SPEA2-PO-Fair,NSGAIII-PO-Fair,RandomSearch-PO-Fair,NSGAII-PO-Fair-Enhanced}" \
  --maximum-actual-tokens "${BUDGET:-5000000}" \
  --max-output-tokens "${MAX_OUTPUT_TOKENS:-16}" \
  --evaluation-split "${EVALUATION_SPLIT:-validation}"
