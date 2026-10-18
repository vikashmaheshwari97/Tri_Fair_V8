#!/usr/bin/env bash
#SBATCH --job-name=tfv82-pilot
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --exclude=falcon[1-6],pegasus,shepard3
#SBATCH --hint=nomultithread
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --array=0-2%3
#SBATCH --output=logs/pilot100k_%A_%a.out
#SBATCH --error=logs/pilot100k_%A_%a.err
set -Eeuo pipefail
module purge
module --ignore_cache load python/3.11.14
module --ignore_cache load python-venv/1.0
source "${TF_VENV:-$HOME/venvs/tri-fair-v8}/bin/activate"
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
source jobs/lib/common.sh
tf_require_research_gpu_node
tf_configure_cpp_runtime
source jobs/lib/model_cache.sh
model="${MODEL:-qwen-3-30b}"
tf_configure_model_cache "$model"
python -m tri_fair.cli.doctor --gpu
datasets=(bbq civil_comments bias_in_bios)
index="${SLURM_ARRAY_TASK_ID:?Submit as the three-task array}"
if [[ "$index" -lt 0 || "$index" -gt 2 ]]; then
    echo "Invalid dataset index: $index" >&2
    exit 2
fi
python -m tri_fair.cli.pilot --execute --model "$model" \
    --dataset "${datasets[$index]}" --seed "${SEED:-72}" --budget 100000 \
    --manifest-dir "${MANIFEST_DIR:-data/splits_v8_clean}"
