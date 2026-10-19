#!/usr/bin/env bash
#SBATCH --job-name=tfv86-report
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/pilot86_report_%j.out
#SBATCH --error=logs/pilot86_report_%j.err
set -Eeuo pipefail
module purge
module --ignore_cache load python/3.11.14
module --ignore_cache load python-venv/1.0
source "${TF_VENV:-$HOME/venvs/tri-fair-v8}/bin/activate"
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
source jobs/lib/pilot_v86_matrix.sh
tf_v86_matrix
base="${PILOT_RESULTS_ROOT:?Shared batch directory required}"
model="${MODEL:-qwen-3-30b}"
split="${EVALUATION_SPLIT:-validation}"
files=()
for seed in "${tf_v86_seeds[@]}"; do
    for dataset in "${tf_v86_datasets[@]}"; do
        file="$base/heldout_$split/$model/$dataset/seed$seed/eval_checkpoints_union.parquet"
        [[ -s "$file" ]] || { echo "Missing held-out result: $file" >&2; exit 1; }
        files+=("$file")
    done
done
out="$base/aggregate_$split/$model"
python -m tri_fair.reporting.multiseed_report \
    --evaluations "${files[@]}" --seeds "${PILOT_SEEDS:-${SEED:-72}}" \
    --methods "${PILOT_METHODS:-Tri-Fair-v8.6,NSGAII-PO-Fair}" \
    --model "$model" --split "$split" --budget "${PILOT_BUDGET:-2000000}" \
    --output-dir "$out"
echo "Aggregated tables and figures: $out"
