#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


from align_m3ed_result_to_gt import load_m3ed_trajectory
from event_slam.core.geometry import invert_transform


def main() -> None:
    args = parse_args()
    estimate = load_m3ed_trajectory(args.estimate)
    ground_truth = load_m3ed_trajectory(args.gt)
    if len(estimate) != len(ground_truth) or not np.allclose(
        estimate.timestamps, ground_truth.timestamps, atol=1e-6, rtol=0.0
    ):
        raise ValueError("Estimate and GT timestamps must match one-to-one")

    position_errors = np.linalg.norm(estimate.positions - ground_truth.positions, axis=1)
    rotation_errors = np.asarray(
        [
            rotation_angle_deg(gt.pose.R.T @ est.pose.R)
            for est, gt in zip(estimate.samples, ground_truth.samples)
        ]
    )
    relative_translation = []
    relative_rotation = []
    for index in range(len(estimate) - 1):
        estimate_delta = invert_transform(estimate.samples[index].pose.as_matrix()) @ estimate.samples[index + 1].pose.as_matrix()
        gt_delta = invert_transform(ground_truth.samples[index].pose.as_matrix()) @ ground_truth.samples[index + 1].pose.as_matrix()
        error = invert_transform(gt_delta) @ estimate_delta
        relative_translation.append(np.linalg.norm(error[:3, 3]))
        relative_rotation.append(rotation_angle_deg(error[:3, :3]))

    gt_length = path_length(ground_truth.positions)
    estimate_length = path_length(estimate.positions)
    lines = (
        f"samples: {len(estimate)}",
        f"duration_s: {estimate.timestamps[-1] - estimate.timestamps[0]:.6f}",
        f"ATE_mean_m: {np.mean(position_errors):.9f}",
        f"ATE_RMSE_m: {np.sqrt(np.mean(position_errors ** 2)):.9f}",
        f"ATE_median_m: {np.median(position_errors):.9f}",
        f"ATE_max_m: {np.max(position_errors):.9f}",
        f"rotation_mean_deg: {np.mean(rotation_errors):.9f}",
        f"rotation_RMSE_deg: {np.sqrt(np.mean(rotation_errors ** 2)):.9f}",
        f"rotation_median_deg: {np.median(rotation_errors):.9f}",
        f"rotation_max_deg: {np.max(rotation_errors):.9f}",
        f"RPE_translation_RMSE_m: {np.sqrt(np.mean(np.square(relative_translation))):.9f}",
        f"RPE_rotation_RMSE_deg: {np.sqrt(np.mean(np.square(relative_rotation))):.9f}",
        f"estimated_path_length_m: {estimate_length:.9f}",
        f"gt_path_length_m: {gt_length:.9f}",
        f"path_length_ratio: {estimate_length / gt_length:.9f}",
    )
    report = "M3ED trajectory metrics\n" + "\n".join(lines)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"saved: {args.output}")


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def path_length(positions: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an aligned M3ED trajectory.")
    parser.add_argument("--estimate", required=True, type=Path)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
