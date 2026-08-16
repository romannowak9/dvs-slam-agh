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
from trajectory_metrics import compute_pose_metrics, match_timestamps


@dataclass
class EvSlamMetrics:
    """
    EvSLAM trajectory evaluation metrics.
    """

    sample_count: int
    ate: float
    ate_rmse: float
    ate_median: float
    ate_max: float
    auc: float
    auc_normalized: float
    mean_rve: float
    median_rve: float
    max_rve: float
    rve_sample_count: int
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
    estimate_indices, gt_indices = match_timestamps(
        estimate[:, 0],
        ground_truth[:, 0],
        args.timestamp_tolerance,
    )
    estimate = estimate[estimate_indices]
    ground_truth = ground_truth[gt_indices]

    metrics = compute_metrics(
        estimate=estimate,
        ground_truth=ground_truth,
        xi_min=args.xi_min,
        xi_max=args.xi_max,
        xi_count=args.xi_count,
        min_speed=args.min_speed,
        rpe_delta_seconds=args.rpe_delta_seconds,
    )

    write_metrics(
        metrics=metrics,
        output_path=args.output,
        estimate_path=args.estimate,
        ground_truth_path=args.gt,
        xi_min=args.xi_min,
        xi_max=args.xi_max,
        xi_count=args.xi_count,
        rpe_delta_seconds=args.rpe_delta_seconds,
        estimate_count=estimate_count,
        gt_count=gt_count,
    )

    print_metrics(metrics)
    print()
    print(f"Saved metrics: {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute EvSLAM ATE and velocity AUC metrics."
    )
    parser.add_argument("--estimate", required=True, type=Path)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument(
        "--output",
        default=Path("outputs/evslam_metrics.txt"),
        type=Path,
    )
    parser.add_argument("--xi-min", default=0.0, type=float)
    parser.add_argument("--xi-max", default=1.0, type=float)
    parser.add_argument("--xi-count", default=1001, type=int)
    parser.add_argument("--min-speed", default=1e-6, type=float)
    parser.add_argument("--timestamp-tolerance", default=1e-6, type=float)
    parser.add_argument("--rpe-delta-seconds", default=1.0, type=float)
    return parser.parse_args()


def compute_metrics(
    estimate: np.ndarray,
    ground_truth: np.ndarray,
    xi_min: float,
    xi_max: float,
    xi_count: int,
    min_speed: float,
    rpe_delta_seconds: float,
) -> EvSlamMetrics:
    """
    Compute ATE and speed-weighted RVE AUC.
    """
    pose_metrics = compute_pose_metrics(
        Trajectory.from_tum_array(estimate[:, :8]),
        Trajectory.from_tum_array(ground_truth[:, :8]),
        rpe_delta_seconds,
    )

    velocity_error = np.linalg.norm(
        estimate[:, 8:11] - ground_truth[:, 8:11],
        axis=1,
    )

    gt_speed = np.linalg.norm(ground_truth[:, 8:11], axis=1)
    rve = velocity_error / np.maximum(gt_speed, float(min_speed))
    moving_rve = rve[gt_speed > float(min_speed)]
    if len(moving_rve) == 0:
        raise ValueError("Cannot compute RVE because all GT speeds are near zero")

    thresholds = np.linspace(float(xi_min), float(xi_max), int(xi_count))
    success = compute_weighted_success_curve(
        rve=rve,
        gt_speed=gt_speed,
        thresholds=thresholds,
    )

    auc = float(np.trapz(success, thresholds))

    interval = float(xi_max - xi_min)
    if interval <= 0.0:
        raise ValueError(f"xi_max must be greater than xi_min, got {xi_min}, {xi_max}")

    auc_normalized = auc / interval

    return EvSlamMetrics(
        sample_count=int(estimate.shape[0]),
        ate=pose_metrics.position_mean,
        ate_rmse=pose_metrics.position_rmse,
        ate_median=pose_metrics.position_median,
        ate_max=pose_metrics.position_max,
        auc=auc,
        auc_normalized=float(auc_normalized),
        mean_rve=float(np.mean(moving_rve)),
        median_rve=float(np.median(moving_rve)),
        max_rve=float(np.max(moving_rve)),
        rve_sample_count=len(moving_rve),
        rotation_rmse_deg=pose_metrics.rotation_rmse_deg,
        rpe_pair_count=pose_metrics.rpe_pair_count,
        rpe_delta_median_s=pose_metrics.rpe_delta_median_s,
        rpe_translation_rmse=pose_metrics.rpe_translation_rmse,
        rpe_rotation_rmse_deg=pose_metrics.rpe_rotation_rmse_deg,
    )


def compute_weighted_success_curve(
    rve: np.ndarray,
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
        mask = rve < threshold
        success[index] = float(np.sum(gt_speed[mask]) / weight_sum)

    return success


def write_metrics(
    metrics: EvSlamMetrics,
    output_path: Path,
    estimate_path: Path,
    ground_truth_path: Path,
    xi_min: float,
    xi_max: float,
    xi_count: int,
    rpe_delta_seconds: float,
    estimate_count: int,
    gt_count: int,
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
        file.write(f"estimate_coverage: {metrics.sample_count / estimate_count:.9f}\n")
        file.write(f"gt_coverage: {metrics.sample_count / gt_count:.9f}\n")
        file.write("\n")

        file.write("Position metrics\n")
        file.write("-" * 80 + "\n")
        file.write(f"ATE: {metrics.ate:.9f}\n")
        file.write(f"ATE_RMSE: {metrics.ate_rmse:.9f}\n")
        file.write(f"ATE_median: {metrics.ate_median:.9f}\n")
        file.write(f"ATE_max: {metrics.ate_max:.9f}\n")
        file.write(f"rotation_RMSE_deg: {metrics.rotation_rmse_deg:.9f}\n")
        file.write(f"RPE_delta_target_s: {rpe_delta_seconds:.9f}\n")
        file.write(f"RPE_delta_median_s: {metrics.rpe_delta_median_s:.9f}\n")
        file.write(f"RPE_pair_count: {metrics.rpe_pair_count}\n")
        file.write(f"RPE_translation_RMSE_m: {metrics.rpe_translation_rmse:.9f}\n")
        file.write(f"RPE_rotation_RMSE_deg: {metrics.rpe_rotation_rmse_deg:.9f}\n")
        file.write("\n")

        file.write("Velocity metrics\n")
        file.write("-" * 80 + "\n")
        file.write(f"xi_min: {xi_min:.9f}\n")
        file.write(f"xi_max: {xi_max:.9f}\n")
        file.write(f"xi_count: {xi_count}\n")
        file.write(f"RVE_sample_count: {metrics.rve_sample_count}\n")
        file.write(f"AUC: {metrics.auc:.9f}\n")
        file.write(f"AUC_normalized: {metrics.auc_normalized:.9f}\n")
        file.write(f"mean_RVE: {metrics.mean_rve:.9f}\n")
        file.write(f"median_RVE: {metrics.median_rve:.9f}\n")
        file.write(f"max_RVE: {metrics.max_rve:.9f}\n")


def print_metrics(metrics: EvSlamMetrics) -> None:
    print("EvSLAM metrics")
    print("=" * 80)
    print(f"sample_count: {metrics.sample_count}")
    print(f"ATE: {metrics.ate:.9f}")
    print(f"ATE_RMSE: {metrics.ate_rmse:.9f}")
    print(f"RPE_translation_RMSE_m: {metrics.rpe_translation_rmse:.9f}")
    print(f"RPE_rotation_RMSE_deg: {metrics.rpe_rotation_rmse_deg:.9f}")
    print(f"AUC: {metrics.auc:.9f}")
    print(f"AUC_normalized: {metrics.auc_normalized:.9f}")
    print(f"median_RVE: {metrics.median_rve:.9f}")


if __name__ == "__main__":
    main()
