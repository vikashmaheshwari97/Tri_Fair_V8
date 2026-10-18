#!/usr/bin/env bash
#SBATCH --job-name=tfv841-valid
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --exclude=falcon[1-6],pegasus,shepard3
#SBATCH --hint=nomultithread
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/pilot_eval_%A_%a.out
#SBATCH --error=logs/pilot_eval_%A_%a.err
set -Eeuo pipefail
module purge
module --ignore_cache load python/3.11.14
module --ignore_cache load python-venv/1.0
source "${TF_VENV:-$HOME/venvs/tri-fair-v8}/bin/activate"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source jobs/lib/common.sh
tf_require_research_gpu_node
tf_configure_cpp_runtime
source jobs/lib/model_cache.sh
model="${MODEL:-qwen-3-30b}"
tf_configure_model_cache "$model"
datasets=(bias_in_bios winobias)
index="${SLURM_ARRAY_TASK_ID:?Submit as an array}"
(( index >= 0 && index < ${#datasets[@]} )) || tf_die "Invalid pilot evaluation index"
dataset="${datasets[$index]}"
seed="${SEED:-72}"
split="${EVALUATION_SPLIT:-validation}"
[[ "$split" == "validation" || "$split" == "test" ]] || tf_die "Invalid evaluation split"
base="${PILOT_RESULTS_ROOT:-results/tri_fair_v841_pilot_1000000}"
study_root="$base/$model"
if [[ -n "${PILOT_STAMP:-}" ]]; then
    study_root="$study_root/$dataset/seed$seed/$PILOT_STAMP"
fi
out="$base/heldout_$split/$model/$dataset/seed$seed"
mkdir -p "$out"
python -m tri_fair.cli.evaluate \
    --study-root "$study_root" --study-profile v8 \
    --model "$model" --dataset "$dataset" --random-seed "$seed" \
    --optimizers "${PILOT_METHODS:-Tri-Fair-v8.4-Hybrid,Tri-Fair-v8.3,NSGAII-PO-Fair}" \
    --manifest-dir "${MANIFEST_DIR:-data/splits_v8_clean}" \
    --maximum-actual-tokens "${PILOT_BUDGET:-1000000}" \
    --max-output-tokens 16 --structured-labels --final-only \
    --evaluation-split "$split" \
    --output-file "$out/eval_checkpoints_union.parquet"
python -m tri_fair.reporting.front_report \
    --evaluations "$out/eval_checkpoints_union.parquet" \
    --split "$split" --budget "${PILOT_BUDGET:-1000000}" \
    --output-dir "$out/indicators"
