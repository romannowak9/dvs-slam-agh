#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))

from event_slam.core.geometry import Pose, invert_transform, make_transform
from event_slam.core.trajectory import Trajectory
from event_slam.core.velocity import VelocityTrajectory
from event_slam.io.result_io import load_evslam_result, write_trajectory_at_timestamps


def main() -> None:
    args = parse_args()

    estimate, estimate_velocity = load_evslam_result(args.estimate)
    gt, _ = load_evslam_result(args.gt)

    estimate_indices, gt_indices = match_timestamps(
        estimate.timestamps,
        gt.timestamps,
        tolerance=args.timestamp_tolerance,
    )

    if len(estimate_indices) < 3:
        raise ValueError(
            "Need at least 3 matched timestamps for trajectory alignment, "
            f"got {len(estimate_indices)}"
        )

    scale = 1.0

    if args.method == "se3":
        T_gt_est = estimate_se3_alignment(
            source_points=estimate.positions[estimate_indices],
            target_points=gt.positions[gt_indices],
        )
    elif args.method == "sim3":
        scale, T_gt_est = estimate_sim3_alignment(
            source_points=estimate.positions[estimate_indices],
            target_points=gt.positions[gt_indices],
        )
    elif args.method == "first_pose":
        T_gt_est = estimate_first_pose_alignment(
            estimate=estimate,
            gt=gt,
            estimate_index=int(estimate_indices[0]),
            gt_index=int(gt_indices[0]),
        )
    else:
        raise ValueError(f"Unsupported method: {args.method}")

    aligned = apply_world_alignment(
        trajectory=estimate,
        T_gt_est=T_gt_est,
        scale=scale,
    )
    aligned_velocity = apply_velocity_alignment(
        trajectory=aligned,
        velocity=estimate_velocity,
        T_gt_est=T_gt_est,
        scale=scale,
    )

    write_trajectory_at_timestamps(
        trajectory=aligned,
        velocity=aligned_velocity,
        timestamps=estimate.timestamps,
        output_path=args.output,
        skip_out_of_range=False,
    )

    rmse_before = position_rmse(
        estimate.positions[estimate_indices],
        gt.positions[gt_indices],
    )
    rmse_after = position_rmse(
        aligned.positions[estimate_indices],
        gt.positions[gt_indices],
    )

    print("Alignment finished")
    print(f"estimate: {args.estimate}")
    print(f"gt:       {args.gt}")
    print(f"output:   {args.output}")
    print(f"method:   {args.method}")
    print(f"matched timestamps: {len(estimate_indices)}")
    print()
    print("T_gt_est:")
    print(T_gt_est)
    print()
    print("det(R_gt_est):", np.linalg.det(T_gt_est[:3, :3]))
    print("scale:", scale)
    print()
    print(f"position RMSE before: {rmse_before:.6f} m")
    print(f"position RMSE after:  {rmse_after:.6f} m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align an EvSLAM estimated trajectory to ground truth."
    )
    parser.add_argument("--estimate", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=["se3", "sim3", "first_pose"],
        default="se3",
    )
    parser.add_argument("--timestamp-tolerance", type=float, default=1e-3)
    return parser.parse_args()


def match_timestamps(
    estimate_timestamps: np.ndarray,
    gt_timestamps: np.ndarray,
    tolerance: float,
) -> tuple:
    estimate_indices = []
    gt_indices = []

    estimate_index = 0
    gt_index = 0

    while estimate_index < len(estimate_timestamps) and gt_index < len(gt_timestamps):
        dt = estimate_timestamps[estimate_index] - gt_timestamps[gt_index]

        if abs(dt) <= tolerance:
            estimate_indices.append(estimate_index)
            gt_indices.append(gt_index)
            estimate_index += 1
            gt_index += 1
        elif dt < 0.0:
            estimate_index += 1
        else:
            gt_index += 1

    return (
        np.asarray(estimate_indices, dtype=np.int64),
        np.asarray(gt_indices, dtype=np.int64),
    )


def estimate_se3_alignment(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> np.ndarray:
    """
    Estimate T_target_source from corresponding positions:

        p_target ~= R_target_source @ p_source + t_target_source

    This is Kabsch alignment without scale.
    """
    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)

    if source_points.shape != target_points.shape:
        raise ValueError(
            "source_points and target_points must have the same shape, "
            f"got {source_points.shape} and {target_points.shape}"
        )

    if source_points.ndim != 2 or source_points.shape[1] != 3:
        raise ValueError(
            f"Expected point arrays with shape (N, 3), got {source_points.shape}"
        )

    source_mean = np.mean(source_points, axis=0)
    target_mean = np.mean(target_points, axis=0)

    source_centered = source_points - source_mean
    target_centered = target_points - target_mean

    H = source_centered.T @ target_centered

    U, _, Vt = np.linalg.svd(H)

    R_target_source = Vt.T @ U.T

    if np.linalg.det(R_target_source) < 0.0:
        Vt[-1, :] *= -1.0
        R_target_source = Vt.T @ U.T

    t_target_source = target_mean - R_target_source @ source_mean

    return make_transform(
        R=R_target_source,
        t=t_target_source,
    )


def estimate_sim3_alignment(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> tuple:
    """
    Estimate Sim(3) alignment from corresponding positions:

        p_target ~= scale * R_target_source @ p_source + t_target_source

    This is Umeyama alignment with scale.
    """
    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)

    if source_points.shape != target_points.shape:
        raise ValueError(
            "source_points and target_points must have the same shape, "
            f"got {source_points.shape} and {target_points.shape}"
        )

    if source_points.ndim != 2 or source_points.shape[1] != 3:
        raise ValueError(
            f"Expected point arrays with shape (N, 3), got {source_points.shape}"
        )

    source_mean = np.mean(source_points, axis=0)
    target_mean = np.mean(target_points, axis=0)

    source_centered = source_points - source_mean
    target_centered = target_points - target_mean

    H = source_centered.T @ target_centered

    U, singular_values, Vt = np.linalg.svd(H)

    D = np.eye(3, dtype=np.float64)

    if np.linalg.det(Vt.T @ U.T) < 0.0:
        D[-1, -1] = -1.0

    R_target_source = Vt.T @ D @ U.T

    source_variance = float(np.sum(source_centered * source_centered))

    if source_variance <= 1e-12:
        raise ValueError("Cannot estimate Sim(3) alignment from degenerate source points")

    scale = float(np.sum(singular_values * np.diag(D)) / source_variance)

    t_target_source = target_mean - scale * (R_target_source @ source_mean)

    T_target_source = make_transform(
        R=R_target_source,
        t=t_target_source,
    )

    return scale, T_target_source


def estimate_first_pose_alignment(
    estimate: Trajectory,
    gt: Trajectory,
    estimate_index: int,
    gt_index: int,
) -> np.ndarray:
    """
    Estimate T_gt_est from the first matched pose:

        T_gt_est = T_Wgt_C0 @ inverse(T_West_C0)
    """
    T_West_C0 = estimate.samples[estimate_index].pose.as_matrix()
    T_Wgt_C0 = gt.samples[gt_index].pose.as_matrix()

    return T_Wgt_C0 @ invert_transform(T_West_C0)


def apply_world_alignment(
    trajectory: Trajectory,
    T_gt_est: np.ndarray,
    scale: float = 1.0,
) -> Trajectory:
    """
    Apply global world-frame alignment:

        R_Wgt_C = R_Wgt_West @ R_West_C
        t_Wgt_C = scale * R_Wgt_West @ t_West_C + t_Wgt_West
    """
    aligned = Trajectory()

    R_gt_est = T_gt_est[:3, :3]
    t_gt_est = T_gt_est[:3, 3]

    for sample in trajectory.samples:
        pose = sample.pose

        R_Wgt_C = R_gt_est @ pose.R
        t_Wgt_C = scale * (R_gt_est @ pose.t) + t_gt_est

        aligned.append(
            timestamp=sample.timestamp,
            pose=Pose(R=R_Wgt_C, t=t_Wgt_C),
        )

    return aligned


def apply_velocity_alignment(
    trajectory: Trajectory,
    velocity: VelocityTrajectory,
    T_gt_est: np.ndarray,
    scale: float = 1.0,
) -> VelocityTrajectory:
    """
    Apply the same world-frame alignment to linear velocity.

    Translation does not affect velocity. For Sim(3), velocity is scaled by the
    same scale as position.
    """
    if len(trajectory) != len(velocity):
        raise ValueError(
            "trajectory and velocity must have the same length, "
            f"got {len(trajectory)} and {len(velocity)}"
        )

    aligned = VelocityTrajectory()
    R_gt_est = T_gt_est[:3, :3]

    for pose_sample, velocity_sample in zip(trajectory.samples, velocity.samples):
        velocity_world = scale * (R_gt_est @ velocity_sample.velocity_world)
        velocity_camera = pose_sample.pose.R.T @ velocity_world

        aligned.append(
            timestamp=pose_sample.timestamp,
            velocity_world=velocity_world,
            velocity_camera=velocity_camera,
        )

    return aligned


def position_rmse(
    estimate_positions: np.ndarray,
    gt_positions: np.ndarray,
) -> float:
    error = np.asarray(estimate_positions) - np.asarray(gt_positions)
    squared_norm = np.sum(error * error, axis=1)

    return float(np.sqrt(np.mean(squared_norm)))


if __name__ == "__main__":
    main()
