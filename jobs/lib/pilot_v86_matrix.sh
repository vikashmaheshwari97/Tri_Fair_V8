#!/usr/bin/env bash
# Shared dataset-major mapping within each seed; usable without Slurm for preflight.
tf_v86_matrix() {
    local -a seen=()
    local raw="${PILOT_SEEDS:-${SEED:-72}}" value prior
    [[ "$raw" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "PILOT_SEEDS must be comma-separated nonnegative integers" >&2; return 2; }
    IFS=',' read -r -a tf_v86_seeds <<< "$raw"
    for value in "${tf_v86_seeds[@]}"; do
        [[ "$value" =~ ^(0|[1-9][0-9]{0,8})$ ]] || { echo "Seeds must be canonical integers below 1e9" >&2; return 2; }
        for prior in "${seen[@]:-}"; do
            [[ "$value" != "$prior" ]] || { echo "Duplicate pilot seed: $value" >&2; return 2; }
        done
        seen+=("$value")
    done
    tf_v86_datasets=(bbq civil_comments bias_in_bios winobias)
    tf_v86_tasks=$((${#tf_v86_seeds[@]} * ${#tf_v86_datasets[@]}))
}
tf_v86_task() {
    local index="${1:?Array index required}"
    [[ "$index" =~ ^(0|[1-9][0-9]{0,8})$ ]] && ((index < tf_v86_tasks)) || { echo "Invalid pilot array index: $index (expected 0-$((tf_v86_tasks-1)))" >&2; return 2; }
    seed="${tf_v86_seeds[$((index / 4))]}"
    dataset="${tf_v86_datasets[$((index % 4))]}"
}
