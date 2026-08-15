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


def main() -> None:
    args = parse_args()

    estimate = load_evslam_result_array(args.estimate)
    ground_truth = load_evslam_result_array(args.gt)

    check_compatible_timestamps(
        estimate=estimate,
        ground_truth=ground_truth,
        tolerance=args.timestamp_tolerance,
    )

    metrics = compute_metrics(
        estimate=estimate,
        ground_truth=ground_truth,
        xi_min=args.xi_min,
        xi_max=args.xi_max,
        xi_count=args.xi_count,
        min_speed=args.min_speed,
    )

    write_metrics(
        metrics=metrics,
        output_path=args.output,
        estimate_path=args.estimate,
        ground_truth_path=args.gt,
        xi_min=args.xi_min,
        xi_max=args.xi_max,
        xi_count=args.xi_count,
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
    return parser.parse_args()


def check_compatible_timestamps(
    estimate: np.ndarray,
    ground_truth: np.ndarray,
    tolerance: float,
) -> None:
    """
    Check that estimated and GT trajectories have matching timestamps.
    """
    if estimate.shape[0] != ground_truth.shape[0]:
        raise ValueError(
            "Estimate and ground truth have different number of samples: "
            f"{estimate.shape[0]} vs {ground_truth.shape[0]}"
        )

    timestamp_error = np.abs(estimate[:, 0] - ground_truth[:, 0])
    max_error = float(np.max(timestamp_error))

    if max_error > tolerance:
        index = int(np.argmax(timestamp_error))
        raise ValueError(
            "Estimate and ground truth timestamps do not match. "
            f"Max error={max_error:.9f} at index={index}, "
            f"estimate={estimate[index, 0]:.9f}, "
            f"gt={ground_truth[index, 0]:.9f}"
        )


def compute_metrics(
    estimate: np.ndarray,
    ground_truth: np.ndarray,
    xi_min: float,
    xi_max: float,
    xi_count: int,
    min_speed: float,
) -> EvSlamMetrics:
    """
    Compute ATE and speed-weighted RVE AUC.
    """
    position_error = np.linalg.norm(
        estimate[:, 1:4] - ground_truth[:, 1:4],
        axis=1,
    )

    velocity_error = np.linalg.norm(
        estimate[:, 8:11] - ground_truth[:, 8:11],
        axis=1,
    )

    gt_speed = np.linalg.norm(ground_truth[:, 8:11], axis=1)
    rve = velocity_error / np.maximum(gt_speed, float(min_speed))

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
        ate=float(np.mean(position_error)),
        ate_rmse=float(np.sqrt(np.mean(position_error ** 2))),
        ate_median=float(np.median(position_error)),
        ate_max=float(np.max(position_error)),
        auc=auc,
        auc_normalized=float(auc_normalized),
        mean_rve=float(np.mean(rve)),
        median_rve=float(np.median(rve)),
        max_rve=float(np.max(rve)),
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
        file.write(f"sample_count: {metrics.sample_count}\n")
        file.write("\n")

        file.write("Position metrics\n")
        file.write("-" * 80 + "\n")
        file.write(f"ATE: {metrics.ate:.9f}\n")
        file.write(f"ATE_RMSE: {metrics.ate_rmse:.9f}\n")
        file.write(f"ATE_median: {metrics.ate_median:.9f}\n")
        file.write(f"ATE_max: {metrics.ate_max:.9f}\n")
        file.write("\n")

        file.write("Velocity metrics\n")
        file.write("-" * 80 + "\n")
        file.write(f"xi_min: {xi_min:.9f}\n")
        file.write(f"xi_max: {xi_max:.9f}\n")
        file.write(f"xi_count: {xi_count}\n")
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
    print(f"AUC: {metrics.auc:.9f}")
    print(f"AUC_normalized: {metrics.auc_normalized:.9f}")
    print(f"median_RVE: {metrics.median_rve:.9f}")


if __name__ == "__main__":
    main()
