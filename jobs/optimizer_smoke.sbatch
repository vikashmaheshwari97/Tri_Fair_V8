#!/usr/bin/env bash
#SBATCH --job-name=tfv8-opt-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --output=logs/optimizer_smoke_%j.out
#SBATCH --error=logs/optimizer_smoke_%j.err
set -Eeuo pipefail
module purge
module --ignore_cache load python/3.11.14
module --ignore_cache load python-venv/1.0
source "${TF_VENV:-$HOME/venvs/tri-fair-v8}/bin/activate"
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
source jobs/lib/common.sh
source jobs/lib/model_cache.sh
model="${MODEL:-qwen-3-30b}"
dataset="${DATASET:-bbq}"
seed="${SEED:-72}"
tf_configure_model_cache "$model"
python -m tri_fair.cli.train --experiment-name optimizer_smoke \
  --model "$model" --dataset "$dataset" --optimizer Tri-Fair-v8 \
  --random-seed "$seed" --n-init-prompts 2 --max-steps 1 \
  --crossovers-per-iteration 1 --max-few-shot-examples 1 \
  --optimizer-overrides configs/smoke.json \
  --budget-per-run 1000000 --budget-checkpoints 1000000 \
  --manifest-dir "${MANIFEST_DIR:-data/splits_v8_clean}" \
  --output-dir "results/optimizer_smoke/$model/$dataset/Tri-Fair-v8/seed$seed" \
  --allow-under-budget-exit
