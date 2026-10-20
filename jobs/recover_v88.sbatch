#!/usr/bin/env bash
#SBATCH --job-name=tfv88-recover
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --exclude=falcon[1-6],pegasus,shepard3
#SBATCH --hint=nomultithread
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=logs/recover88_%j.out
#SBATCH --error=logs/recover88_%j.err
set -Eeuo pipefail
module purge
module --ignore_cache load python/3.11.14
module --ignore_cache load python-venv/1.0
source "${TF_VENV:-$HOME/venvs/tri-fair-v8}/bin/activate"
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
source jobs/lib/common.sh
tf_require_research_gpu_node
tf_configure_cpp_runtime
source jobs/lib/model_cache.sh
tf_configure_model_cache qwen-3-30b
python recovery/v88_recover.py execute \
    --batch-root "${RECOVERY_BATCH:?Original batch required}" \
    --output-root "${PILOT_RESULTS_ROOT:?New recovery output required}"
