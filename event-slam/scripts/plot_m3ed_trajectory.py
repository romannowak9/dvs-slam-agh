#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


from align_m3ed_result_to_gt import load_m3ed_trajectory
from event_slam.core.trajectory import Trajectory
from event_slam.core.velocity import compute_velocity_trajectory
from plot_trajectory import (
    TrajectoryPlotData,
    plot_component,
    plot_trajectory_3d,
)


def main() -> None:
    args = parse_args()
    estimate = load_plot_data(args.trajectory, "estimated")
    ground_truth = trim_to_estimate(
        load_plot_data(args.gt, "ground truth"),
        estimate,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    time_origin = 0.0 if args.absolute_time else estimate.timestamps[0]
    xlabel = "Timestamp [s]" if args.absolute_time else "Time from start [s]"

    components = (
        ("x", estimate.positions[:, 0], ground_truth.positions[:, 0], "Trajectory x(t)", "x [m]"),
        ("y", estimate.positions[:, 1], ground_truth.positions[:, 1], "Trajectory y(t)", "y [m]"),
        ("z", estimate.positions[:, 2], ground_truth.positions[:, 2], "Trajectory z(t)", "z [m]"),
        ("vx", estimate.velocities_camera[:, 0], ground_truth.velocities_camera[:, 0], "Camera velocity vx(t)", "vx [m/s]"),
        ("vy", estimate.velocities_camera[:, 1], ground_truth.velocities_camera[:, 1], "Camera velocity vy(t)", "vy [m/s]"),
        ("vz", estimate.velocities_camera[:, 2], ground_truth.velocities_camera[:, 2], "Camera velocity vz(t)", "vz [m/s]"),
        ("speed", estimate.speeds, ground_truth.speeds, "Camera speed(t)", "speed [m/s]"),
    )
    for suffix, estimate_values, gt_values, title, ylabel in components:
        plot_component(
            estimate,
            ground_truth,
            time_origin,
            estimate_values,
            gt_values,
            title,
            ylabel,
            xlabel,
            args.output_dir / f"{args.prefix}_{suffix}.png",
        )
    plot_trajectory_3d(
        estimate,
        ground_truth,
        args.output_dir / f"{args.prefix}_3d.png",
        axis_order=(0, 2, 1),
        axis_signs=(1.0, -1.0, 1.0),
    )
    print(f"saved 8 plots: {args.output_dir}")


def load_plot_data(path: Path, label: str) -> TrajectoryPlotData:
    trajectory = load_m3ed_trajectory(path)
    velocity = compute_velocity_trajectory(
        trajectory,
        smoothing_window_size=5,
        smoothing_poly_order=2,
    )
    return TrajectoryPlotData(label, trajectory, velocity.velocities_camera)


def trim_to_estimate(
    ground_truth: TrajectoryPlotData,
    estimate: TrajectoryPlotData,
) -> TrajectoryPlotData:
    mask = (
        (ground_truth.timestamps >= estimate.timestamps[0])
        & (ground_truth.timestamps <= estimate.timestamps[-1])
    )
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        raise ValueError("Estimate and ground truth time ranges do not overlap")
    trajectory = Trajectory(
        [ground_truth.trajectory.samples[index] for index in indices]
    )
    return TrajectoryPlotData(
        ground_truth.label,
        trajectory,
        ground_truth.velocities_camera[indices],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot an M3ED trajectory and GT.")
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prefix", default="trajectory")
    parser.add_argument("--absolute-time", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
