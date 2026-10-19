#!/usr/bin/env bash
# One-seed screening with the same accounting and final selection for all methods.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PILOT_SEEDS="${PILOT_SEEDS:-${SEED:-72}}"
export PILOT_METHODS="${PILOT_METHODS:-Tri-Fair-v8.6,NSGAIII-PO-Fair,SPEA2-PO-Fair,RandomSearch-PO-Fair}"
export PILOT_RECLAIM_RESERVE=1
export PILOT_RESULTS_ROOT="${PILOT_RESULTS_ROOT:-results/tri_fair_v86_efficiency_${PILOT_BUDGET:-2000000}/batch_$(date -u +%Y%m%dT%H%M%S)_${RANDOM}}"
exec bash jobs/submit_pilot_v86.sh
