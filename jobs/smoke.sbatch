#!/usr/bin/env bash
#SBATCH --job-name=tfv8-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --exclude=falcon[1-6],pegasus,shepard3
#SBATCH --hint=nomultithread
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err
set -Eeuo pipefail
module purge
module --ignore_cache load python/3.11.14
module --ignore_cache load python-venv/1.0
source "${TF_VENV:-$HOME/venvs/tri-fair-v8}/bin/activate"
cd "${SLURM_SUBMIT_DIR:?Submit this job from the project root}"
source jobs/lib/common.sh
tf_require_research_gpu_node
tf_configure_cpp_runtime
source jobs/lib/model_cache.sh
tf_log "Allocation: job=${SLURM_JOB_ID:-local} nodes=${SLURM_JOB_NODELIST:-unknown}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -m tri_fair.cli.doctor --gpu
tf_configure_model_cache "${MODEL:-qwen-3-30b}"
python -m tri_fair.cli.smoke --model "${MODEL:-qwen-3-30b}" \
  --dataset "${DATASET:-bbq}" --random-seed "${SEED:-72}" \
  --manifest-dir "${MANIFEST_DIR:-data/splits_v8_clean}"
