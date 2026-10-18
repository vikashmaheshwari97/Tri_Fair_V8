#!/usr/bin/env bash
#SBATCH --job-name=tfv841-pilot
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --exclude=falcon[1-6],pegasus,shepard3
#SBATCH --hint=nomultithread
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%2
#SBATCH --output=logs/pilot_%A_%a.out
#SBATCH --error=logs/pilot_%A_%a.err
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
datasets=(bbq civil_comments bias_in_bios winobias)
index="${SLURM_ARRAY_TASK_ID:?Submit as the four-task array}"
if [[ "$index" -lt 0 || "$index" -gt 3 ]]; then
    echo "Invalid dataset index: $index" >&2
    exit 2
fi
budget="${PILOT_BUDGET:-1000000}"
if [[ ! "$budget" =~ ^[1-9][0-9]*$ ]]; then
    echo "PILOT_BUDGET must be a positive integer" >&2
    exit 2
fi
pilot_args=(--execute --model "$model" --dataset "${datasets[$index]}"
    --seed "${SEED:-72}" --budget "$budget"
    --n-init-prompts "${PILOT_INIT_PROMPTS:-2}"
    --max-few-shot-examples "${PILOT_MAX_SHOTS:-1}"
    --crossovers-per-iteration "${PILOT_CHALLENGERS:-1}"
    --final-search-candidates "${PILOT_AUDIT_CANDIDATES:-2}"
    --final-search-reserve "${PILOT_AUDIT_RESERVE:-0.35}"
    --manifest-dir "${MANIFEST_DIR:-data/splits_v8_clean}")
pilot_args+=(--methods "${PILOT_METHODS:-Tri-Fair-v8.4-Hybrid,Tri-Fair-v8.3,NSGAII-PO-Fair}")
tf_log "Pilot: model=$model dataset=${datasets[$index]} cap=$budget (search and final audit combined)"
if [[ "${PILOT_STRUCTURED_LABELS:-1}" == "0" ]]; then
    pilot_args+=(--unconstrained-labels)
fi
python -m tri_fair.cli.pilot "${pilot_args[@]}"
