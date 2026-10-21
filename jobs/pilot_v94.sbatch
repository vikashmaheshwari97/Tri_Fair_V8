#!/usr/bin/env bash
#SBATCH --job-name=tfv94-three
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --exclude=falcon[1-6],pegasus,shepard3
#SBATCH --hint=nomultithread
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%5
#SBATCH --output=logs/pilot94_%A_%a.out
#SBATCH --error=logs/pilot94_%A_%a.err
set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
source jobs/lib/activate_existing_venv.sh
tf_activate_existing_venv
# Export before the first vLLM/torch import, consistently for all five methods.
export VLLM_BATCH_INVARIANT=1 MPLBACKEND=Agg
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
source jobs/lib/common.sh
tf_require_research_gpu_node
tf_configure_cpp_runtime
source jobs/lib/model_cache.sh
tf_configure_model_cache "${MODEL:-qwen-3-30b}"
python research/run_pilot_v94.py "${TF_MATRIX_STAGE:?train or evaluate required}" --index "${SLURM_ARRAY_TASK_ID:?Array required}"
