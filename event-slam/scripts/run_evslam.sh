#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

config="${1:-configs/evslam.yaml}"
paths="$(python3 scripts/config_run_paths.py --dataset evslam "$config")"
mapfile -t run_paths <<< "$paths"
output="${run_paths[0]}"
raw="${run_paths[1]}"
prefix="${run_paths[2]}"
gt="${2:-${run_paths[3]:-}}"
aligned="$output/result_aligned_first_pose.txt"

python3 scripts/run_event_slam.py --config "$config"
if [[ -z "$gt" ]]; then
    echo "Ground truth not configured; skipping alignment, plots and metrics."
    exit 0
fi
python3 scripts/align_evslam_result_to_gt.py --estimate "$raw" --gt "$gt" --output "$aligned" --method first_pose
python3 scripts/plot_trajectory.py --trajectory "$aligned" --gt "$gt" --output-dir "$output/plots_first_pose" --prefix "${prefix}_first_pose"
python3 scripts/evaluate_evslam_metrics.py --estimate "$aligned" --gt "$gt" --output "$output/metrics_first_pose.txt" --alignment none
