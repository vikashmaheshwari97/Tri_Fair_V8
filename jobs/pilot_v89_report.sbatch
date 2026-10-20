#!/usr/bin/env bash
#SBATCH --job-name=tfv89-report
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/pilot89_report_%j.out
#SBATCH --error=logs/pilot89_report_%j.err
set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
source jobs/lib/activate_existing_venv.sh
tf_activate_existing_venv
export MPLBACKEND=Agg
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python research/run_pilot_v89.py report
