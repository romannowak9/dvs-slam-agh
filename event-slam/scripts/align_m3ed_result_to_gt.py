#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


from align_evslam_result_to_gt import (
    apply_world_alignment,
    estimate_first_pose_alignment,
    estimate_se3_alignment,
    estimate_sim3_alignment,
    match_timestamps,
    position_rmse,
)
from event_slam.core.trajectory import Trajectory
from event_slam.io.result_io import (
    trajectory_from_m3ed_array,
    trajectory_to_m3ed_array,
)


def main() -> None:
    args = parse_args()
    estimate = load_m3ed_trajectory(args.estimate)
    ground_truth = load_m3ed_trajectory(args.gt)
    estimate_indices, gt_indices = match_timestamps(
        estimate.timestamps,
        ground_truth.timestamps,
        tolerance=args.timestamp_tolerance,
    )
    if len(estimate_indices) < 3:
        raise ValueError(f"Need at least 3 matched poses, got {len(estimate_indices)}")

    scale = 1.0
    if args.method == "first_pose":
        T_gt_est = estimate_first_pose_alignment(
            estimate, ground_truth, estimate_indices[0], gt_indices[0]
        )
    elif args.method == "se3":
        T_gt_est = estimate_se3_alignment(
            estimate.positions[estimate_indices],
            ground_truth.positions[gt_indices],
        )
    else:
        scale, T_gt_est = estimate_sim3_alignment(
            estimate.positions[estimate_indices],
            ground_truth.positions[gt_indices],
        )

    aligned = apply_world_alignment(estimate, T_gt_est, scale)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, trajectory_to_m3ed_array(aligned), fmt="%.9f")
    before = position_rmse(
        estimate.positions[estimate_indices], ground_truth.positions[gt_indices]
    )
    after = position_rmse(
        aligned.positions[estimate_indices], ground_truth.positions[gt_indices]
    )
    print(f"method: {args.method}")
    print(f"matched_timestamps: {len(estimate_indices)}")
    print(f"scale: {scale:.9f}")
    print(f"position_RMSE_before_m: {before:.6f}")
    print(f"position_RMSE_after_m: {after:.6f}")
    print(f"saved: {args.output}")


def load_m3ed_trajectory(path: Path) -> Trajectory:
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2 or data.shape[1] != 8:
        raise ValueError(f"Expected an M3ED Nx8 trajectory, got {data.shape}: {path}")
    return trajectory_from_m3ed_array(data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align an M3ED trajectory to GT.")
    parser.add_argument("--estimate", required=True, type=Path)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--method", choices=("first_pose", "se3", "sim3"), default="se3"
    )
    parser.add_argument("--timestamp-tolerance", type=float, default=1e-6)
    return parser.parse_args()


if __name__ == "__main__":
    main()
