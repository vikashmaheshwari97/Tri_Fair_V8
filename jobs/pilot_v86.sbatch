#!/usr/bin/env bash
#SBATCH --job-name=tfv86-pilot
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
#SBATCH --output=logs/pilot86_%A_%a.out
#SBATCH --error=logs/pilot86_%A_%a.err
set -Eeuo pipefail
module purge
module --ignore_cache load python/3.11.14
module --ignore_cache load python-venv/1.0
source "${TF_VENV:-$HOME/venvs/tri-fair-v8}/bin/activate"
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
source jobs/lib/common.sh
source jobs/lib/pilot_v86_matrix.sh
tf_v86_matrix
tf_v86_task "${SLURM_ARRAY_TASK_ID:?Submit as an array}"
tf_require_research_gpu_node
tf_configure_cpp_runtime
source jobs/lib/model_cache.sh
model="${MODEL:-qwen-3-30b}"
tf_configure_model_cache "$model"
python -m tri_fair.cli.doctor --gpu
budget="${PILOT_BUDGET:-2000000}"
if [[ ! "$budget" =~ ^[1-9][0-9]*$ ]]; then
    echo "PILOT_BUDGET must be a positive integer" >&2
    exit 2
fi
pilot_args=(--execute --model "$model" --dataset "$dataset"
    --seed "$seed" --budget "$budget"
    --n-init-prompts "${PILOT_INIT_PROMPTS:-4}"
    --max-few-shot-examples "${PILOT_MAX_SHOTS:-2}"
    --crossovers-per-iteration "${PILOT_CHALLENGERS:-1}"
    --final-search-candidates "${PILOT_AUDIT_CANDIDATES:-4}"
    --final-search-reserve "${PILOT_AUDIT_RESERVE:-0.30}"
    --manifest-dir "${MANIFEST_DIR:-data/splits_v8_clean}")
pilot_args+=(--output-root "${PILOT_RESULTS_ROOT:-results/tri_fair_v86_pilot_$budget}")
pilot_args+=(--methods "${PILOT_METHODS:-Tri-Fair-v8.6,NSGAII-PO-Fair}")
tf_log "Pilot: model=$model dataset=$dataset cap=$budget (search and final audit combined)"
if [[ "${PILOT_STRUCTURED_LABELS:-1}" == "0" ]]; then
    pilot_args+=(--unconstrained-labels)
fi
if [[ "${PILOT_RECLAIM_RESERVE:-0}" == "1" ]]; then
    pilot_args+=(--reclaim-audit-reserve)
fi
python -m tri_fair.cli.pilot_v86 "${pilot_args[@]}"
