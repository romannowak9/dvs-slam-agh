#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))

from event_slam.io.result_io import load_evslam_result_array
from event_slam.core.trajectory import Trajectory
from align_evslam_result_to_gt import apply_world_alignment, estimate_se3_alignment
from trajectory_metrics import compute_pose_metrics

EVSLAM_XI_MIN = 0.0
EVSLAM_XI_MAX = 1.0
EVSLAM_XI_COUNT = 500


@dataclass
class EvSlamMetrics:
    """
    EvSLAM trajectory evaluation metrics.
    """

    sample_count: int
    position_mean: float
    ate_rmse: float
    position_median: float
    position_max: float
    auc: float
    rotation_rmse_deg: float
    rpe_pair_count: int
    rpe_delta_median_s: float
    rpe_translation_rmse: float
    rpe_rotation_rmse_deg: float


def main() -> None:
    args = parse_args()

    estimate = load_evslam_result_array(args.estimate)
    ground_truth = load_evslam_result_array(args.gt)

    estimate_count = len(estimate)
    gt_count = len(ground_truth)
    estimate_indices, gt_indices = match_evslam_timestamps(
        estimate[:, 0],
        ground_truth[:, 0],
        args.timestamp_tolerance,
    )
    estimate = estimate[estimate_indices]
    ground_truth = ground_truth[gt_indices]

    metrics = compute_metrics(
        estimate=estimate,
        ground_truth=ground_truth,
        rpe_delta_seconds=args.rpe_delta_seconds,
        alignment=args.alignment,
    )

    write_metrics(
        metrics=metrics,
        output_path=args.output,
        estimate_path=args.estimate,
        ground_truth_path=args.gt,
        rpe_delta_seconds=args.rpe_delta_seconds,
        estimate_count=estimate_count,
        gt_count=gt_count,
        alignment=args.alignment,
        timestamp_tolerance=args.timestamp_tolerance,
    )

    print_metrics(metrics, args.alignment)
    print()
    print(f"Saved metrics: {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute EvSLAM ATE, AUC, RMSE, and RPE with the official protocol."
    )
    parser.add_argument("--estimate", required=True, type=Path)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument(
        "--output",
        default=Path("outputs/evslam_metrics.txt"),
        type=Path,
    )
    parser.add_argument("--timestamp-tolerance", default=1e-3, type=float)
    parser.add_argument("--rpe-delta-seconds", default=1.0, type=float)
    parser.add_argument(
        "--alignment",
        choices=("se3", "none"),
        default="se3",
        help="Use official rigid Umeyama alignment or evaluate an already aligned trajectory.",
    )
    return parser.parse_args()


def compute_metrics(
    estimate: np.ndarray,
    ground_truth: np.ndarray,
    rpe_delta_seconds: float,
    alignment: str,
) -> EvSlamMetrics:
    """
    Compute pose errors and the speed-weighted AUC used by EvSLAM.
    """
    estimate_trajectory = Trajectory.from_tum_array(estimate[:, :8])
    ground_truth_trajectory = Trajectory.from_tum_array(ground_truth[:, :8])
    if alignment == "se3":
        # The official evaluator serializes matched poses with six decimal
        # places before applying rigid Umeyama alignment.  Reproduce that
        # detail so ATE agrees down to its numerical precision.
        estimate_positions = np.round(estimate[:, 1:4], decimals=6)
        ground_truth_positions = np.round(ground_truth[:, 1:4], decimals=6)
        T_gt_est = estimate_se3_alignment(
            estimate_positions,
            ground_truth_positions,
        )
        evaluated_trajectory = apply_world_alignment(
            estimate_trajectory,
            T_gt_est,
            scale=1.0,
        )
        R_gt_est = T_gt_est[:3, :3]
        t_gt_est = T_gt_est[:3, 3]
        aligned_positions = (R_gt_est @ estimate_positions.T).T + t_gt_est
        official_position_error = np.linalg.norm(
            aligned_positions - ground_truth_positions,
            axis=1,
        )
        ate_rmse = float(np.sqrt(np.mean(official_position_error ** 2)))
    elif alignment == "none":
        evaluated_trajectory = estimate_trajectory
        position_error = np.linalg.norm(
            estimate_trajectory.positions - ground_truth_trajectory.positions,
            axis=1,
        )
        ate_rmse = float(np.sqrt(np.mean(position_error ** 2)))
    else:
        raise ValueError(f"Unsupported alignment: {alignment}")

    pose_metrics = compute_pose_metrics(
        evaluated_trajectory,
        ground_truth_trajectory,
        rpe_delta_seconds,
    )

    velocity_error = np.linalg.norm(
        estimate[:, 8:11] - ground_truth[:, 8:11],
        axis=1,
    )

    gt_speed = np.linalg.norm(ground_truth[:, 8:11], axis=1)
    relative_velocity_error = np.zeros_like(gt_speed)
    nonzero_speed = gt_speed != 0.0
    relative_velocity_error[nonzero_speed] = (
        velocity_error[nonzero_speed] / gt_speed[nonzero_speed]
    )

    # The official evaluator writes these intermediate values with eight
    # decimal places before integrating the success curve.
    relative_velocity_error = np.round(relative_velocity_error, decimals=8)
    gt_speed = np.round(gt_speed, decimals=8)

    thresholds = np.linspace(EVSLAM_XI_MIN, EVSLAM_XI_MAX, EVSLAM_XI_COUNT)
    success = compute_weighted_success_curve(
        relative_velocity_error=relative_velocity_error,
        gt_speed=gt_speed,
        thresholds=thresholds,
    )

    auc = float(np.trapz(success, thresholds))

    return EvSlamMetrics(
        sample_count=int(estimate.shape[0]),
        position_mean=pose_metrics.position_mean,
        ate_rmse=ate_rmse,
        position_median=pose_metrics.position_median,
        position_max=pose_metrics.position_max,
        auc=auc,
        rotation_rmse_deg=pose_metrics.rotation_rmse_deg,
        rpe_pair_count=pose_metrics.rpe_pair_count,
        rpe_delta_median_s=pose_metrics.rpe_delta_median_s,
        rpe_translation_rmse=pose_metrics.rpe_translation_rmse,
        rpe_rotation_rmse_deg=pose_metrics.rpe_rotation_rmse_deg,
    )


def match_evslam_timestamps(
    estimate_timestamps: np.ndarray,
    gt_timestamps: np.ndarray,
    tolerance: float,
) -> tuple:
    """Reproduce the nearest-neighbour timestamp matching in evaluate.py."""
    estimate_timestamps = np.asarray(estimate_timestamps, dtype=np.float64)
    gt_timestamps = np.asarray(gt_timestamps, dtype=np.float64)
    estimate_order = np.argsort(estimate_timestamps)
    estimate_sorted = estimate_timestamps[estimate_order]

    estimate_indices = []
    gt_indices = []
    estimate_index = 0
    for gt_index, gt_timestamp in enumerate(gt_timestamps):
        while (
            estimate_index < len(estimate_sorted)
            and estimate_sorted[estimate_index] < gt_timestamp
        ):
            estimate_index += 1

        best_index = None
        best_distance = float("inf")
        if estimate_index < len(estimate_sorted):
            distance = abs(estimate_sorted[estimate_index] - gt_timestamp)
            if distance <= tolerance and distance < best_distance:
                best_index = estimate_index
                best_distance = distance
        if estimate_index > 0:
            distance = abs(estimate_sorted[estimate_index - 1] - gt_timestamp)
            if distance <= tolerance and distance < best_distance:
                best_index = estimate_index - 1

        if best_index is not None:
            estimate_indices.append(int(estimate_order[best_index]))
            gt_indices.append(gt_index)

    if not estimate_indices:
        raise ValueError("Estimate and ground truth have no matching timestamps")
    return (
        np.asarray(estimate_indices, dtype=np.int64),
        np.asarray(gt_indices, dtype=np.int64),
    )


def compute_weighted_success_curve(
    relative_velocity_error: np.ndarray,
    gt_speed: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """
    Compute EvSLAM speed-weighted success curve S_xi.
    """
    weight_sum = float(np.sum(gt_speed))

    if weight_sum <= 0.0:
        raise ValueError("Cannot compute AUC because all GT speeds are zero")

    success = np.empty(len(thresholds), dtype=np.float64)

    for index, threshold in enumerate(thresholds):
        mask = relative_velocity_error < threshold
        success[index] = float(np.sum(gt_speed[mask]) / weight_sum)

    return success


def write_metrics(
    metrics: EvSlamMetrics,
    output_path: Path,
    estimate_path: Path,
    ground_truth_path: Path,
    rpe_delta_seconds: float,
    estimate_count: int,
    gt_count: int,
    alignment: str,
    timestamp_tolerance: float,
) -> None:
    """
    Save metrics to a text file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("EvSLAM metrics\n")
        file.write("=" * 80 + "\n")
        file.write(f"estimate: {estimate_path}\n")
        file.write(f"ground_truth: {ground_truth_path}\n")
        file.write(f"estimate_sample_count: {estimate_count}\n")
        file.write(f"gt_sample_count: {gt_count}\n")
        file.write(f"matched_sample_count: {metrics.sample_count}\n")
        file.write(f"timestamp_tolerance_s: {timestamp_tolerance:.9f}\n")
        file.write(f"alignment: {alignment}\n")
        file.write(f"estimate_coverage: {metrics.sample_count / estimate_count:.9f}\n")
        file.write(f"gt_coverage: {metrics.sample_count / gt_count:.9f}\n")
        file.write("\n")

        file.write("Position metrics\n")
        file.write("-" * 80 + "\n")
        file.write(f"position_error_mean_m: {metrics.position_mean:.9f}\n")
        if alignment == "se3":
            file.write(f"ATE_RMSE_m: {metrics.ate_rmse:.9f}\n")
        else:
            file.write(f"position_RMSE_m: {metrics.ate_rmse:.9f}\n")
        file.write(f"position_error_median_m: {metrics.position_median:.9f}\n")
        file.write(f"position_error_max_m: {metrics.position_max:.9f}\n")
        file.write(f"rotation_RMSE_deg: {metrics.rotation_rmse_deg:.9f}\n")
        file.write(f"RPE_delta_target_s: {rpe_delta_seconds:.9f}\n")
        file.write(f"RPE_delta_median_s: {metrics.rpe_delta_median_s:.9f}\n")
        file.write(f"RPE_pair_count: {metrics.rpe_pair_count}\n")
        file.write(f"RPE_translation_RMSE_m: {metrics.rpe_translation_rmse:.9f}\n")
        file.write(f"RPE_rotation_RMSE_deg: {metrics.rpe_rotation_rmse_deg:.9f}\n")
        file.write("\n")

        file.write("Velocity metrics\n")
        file.write("-" * 80 + "\n")
        file.write(f"xi_min: {EVSLAM_XI_MIN:.9f}\n")
        file.write(f"xi_max: {EVSLAM_XI_MAX:.9f}\n")
        file.write(f"xi_count: {EVSLAM_XI_COUNT}\n")
        file.write(f"AUC: {metrics.auc:.9f}\n")


def print_metrics(metrics: EvSlamMetrics, alignment: str) -> None:
    print("EvSLAM metrics")
    print("=" * 80)
    print(f"sample_count: {metrics.sample_count}")
    print(f"position_error_mean_m: {metrics.position_mean:.9f}")
    if alignment == "se3":
        print(f"ATE_RMSE_m: {metrics.ate_rmse:.9f}")
    else:
        print(f"position_RMSE_m: {metrics.ate_rmse:.9f}")
    print(f"RPE_translation_RMSE_m: {metrics.rpe_translation_rmse:.9f}")
    print(f"RPE_rotation_RMSE_deg: {metrics.rpe_rotation_rmse_deg:.9f}")
    print(f"AUC: {metrics.auc:.9f}")


if __name__ == "__main__":
    main()
