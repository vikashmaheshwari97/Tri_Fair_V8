#!/usr/bin/env bash
#SBATCH --job-name=tfv88-stability
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=h200-141g:1
#SBATCH --exclude=falcon[1-6],pegasus,pegasus2,shepard[2-3]
#SBATCH --hint=nomultithread
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=logs/bios_stable88_%j.out
#SBATCH --error=logs/bios_stable88_%j.err
set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source jobs/lib/activate_existing_venv.sh
tf_activate_existing_venv
# Set this BEFORE any Python process that could import vLLM.
export VLLM_BATCH_INVARIANT=1
source jobs/lib/common.sh
tf_require_research_gpu_node
case "${SLURMD_NODENAME:-$(hostname -s)}" in
  firefly1|firefly2|firefly3) ;;
  *) tf_die 'This stability profile is restricted to the Firefly H200 family.' ;;
esac
tf_configure_cpp_runtime
source jobs/lib/model_cache.sh
tf_configure_model_cache qwen-3-30b
case "${STABILITY_STAGE:?}" in
  probe)
    python recovery/v88_bios_stability.py probe \
      --original-probe "${ORIGINAL_PROBE:?}" --output-root "${PROBE_OUTPUT:?}" \
      --manifest-dir "${MANIFEST_DIR:-data/splits_v8_clean}"
    ;;
  train|evaluate)
    python recovery/v88_bios_stability.py "$STABILITY_STAGE" \
      --proof "${PROBE_PROOF:?}" --output-root "${BIOS_OUTPUT:?}"
    ;;
  *) tf_die 'Unknown stability stage' ;;
esac
