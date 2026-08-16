#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


from align_m3ed_result_to_gt import load_m3ed_trajectory
from trajectory_metrics import (
    compute_pose_metrics,
    match_timestamps,
    select_trajectory_samples,
)


def main() -> None:
    args = parse_args()
    estimate = load_m3ed_trajectory(args.estimate)
    ground_truth = load_m3ed_trajectory(args.gt)
    estimate_count = len(estimate)
    gt_count = len(ground_truth)
    estimate_indices, gt_indices = match_timestamps(
        estimate.timestamps,
        ground_truth.timestamps,
        args.timestamp_tolerance,
    )
    estimate = select_trajectory_samples(estimate, estimate_indices)
    ground_truth = select_trajectory_samples(ground_truth, gt_indices)

    metrics = compute_pose_metrics(
        estimate,
        ground_truth,
        args.rpe_delta_seconds,
    )

    gt_length = path_length(ground_truth.positions)
    estimate_length = path_length(estimate.positions)
    lines = (
        f"estimate_samples: {estimate_count}",
        f"gt_samples: {gt_count}",
        f"matched_samples: {len(estimate)}",
        f"estimate_coverage: {len(estimate) / estimate_count:.9f}",
        f"gt_coverage: {len(estimate) / gt_count:.9f}",
        f"duration_s: {estimate.timestamps[-1] - estimate.timestamps[0]:.6f}",
        f"APE_translation_mean_m: {metrics.position_mean:.9f}",
        f"APE_translation_RMSE_m: {metrics.position_rmse:.9f}",
        f"APE_translation_median_m: {metrics.position_median:.9f}",
        f"APE_translation_max_m: {metrics.position_max:.9f}",
        f"rotation_mean_deg: {metrics.rotation_mean_deg:.9f}",
        f"rotation_RMSE_deg: {metrics.rotation_rmse_deg:.9f}",
        f"rotation_median_deg: {metrics.rotation_median_deg:.9f}",
        f"rotation_max_deg: {metrics.rotation_max_deg:.9f}",
        f"RPE_delta_target_s: {args.rpe_delta_seconds:.9f}",
        f"RPE_delta_median_s: {metrics.rpe_delta_median_s:.9f}",
        f"RPE_pair_count: {metrics.rpe_pair_count}",
        f"RPE_translation_RMSE_m: {metrics.rpe_translation_rmse:.9f}",
        f"RPE_rotation_RMSE_deg: {metrics.rpe_rotation_rmse_deg:.9f}",
        f"estimated_path_length_m: {estimate_length:.9f}",
        f"gt_path_length_m: {gt_length:.9f}",
        f"path_length_ratio: {estimate_length / gt_length:.9f}",
    )
    report = "M3ED trajectory metrics\n" + "\n".join(lines)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"saved: {args.output}")


def path_length(positions: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an aligned M3ED trajectory.")
    parser.add_argument("--estimate", required=True, type=Path)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rpe-delta-seconds", default=1.0, type=float)
    parser.add_argument("--timestamp-tolerance", default=1e-6, type=float)
    return parser.parse_args()


if __name__ == "__main__":
    main()
