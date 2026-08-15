#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))


from event_slam.core.trajectory import Trajectory
from event_slam.io.result_io import load_evslam_result_array


@dataclass
class TrajectoryPlotData:
    """
    Trajectory data loaded from the EvSLAM result format.

    Expected input columns:
        timestamp tx ty tz qx qy qz qw vx vy vz
    """

    label: str
    trajectory: Trajectory
    velocities_camera: np.ndarray

    @property
    def timestamps(self) -> np.ndarray:
        return self.trajectory.timestamps

    @property
    def positions(self) -> np.ndarray:
        return self.trajectory.positions

    @property
    def speeds(self) -> np.ndarray:
        return np.linalg.norm(self.velocities_camera, axis=1)


def main() -> None:
    args = parse_args()

    estimated = load_trajectory_plot_data(
        path=args.trajectory,
        label="estimated",
    )

    ground_truth = None
    if args.gt is not None:
        ground_truth = load_trajectory_plot_data(
            path=args.gt,
            label="ground truth",
        )

    if estimated.trajectory.is_empty:
        raise ValueError(f"Trajectory is empty: {args.trajectory}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.absolute_time:
        time_origin = 0.0
        time_label = "Timestamp [s]"
    else:
        time_origin = estimated.timestamps[0]
        time_label = "Time from start [s]"

    saved_paths = []

    saved_paths.append(
        plot_component(
            estimated=estimated,
            ground_truth=ground_truth,
            time_origin=time_origin,
            estimated_values=estimated.positions[:, 0],
            gt_values=get_gt_values(ground_truth, "position", 0),
            title="Trajectory x(t)",
            ylabel="x [m]",
            xlabel=time_label,
            output_path=output_dir / f"{args.prefix}_x.{args.format}",
        )
    )

    saved_paths.append(
        plot_component(
            estimated=estimated,
            ground_truth=ground_truth,
            time_origin=time_origin,
            estimated_values=estimated.positions[:, 1],
            gt_values=get_gt_values(ground_truth, "position", 1),
            title="Trajectory y(t)",
            ylabel="y [m]",
            xlabel=time_label,
            output_path=output_dir / f"{args.prefix}_y.{args.format}",
        )
    )

    saved_paths.append(
        plot_component(
            estimated=estimated,
            ground_truth=ground_truth,
            time_origin=time_origin,
            estimated_values=estimated.positions[:, 2],
            gt_values=get_gt_values(ground_truth, "position", 2),
            title="Trajectory z(t)",
            ylabel="z [m]",
            xlabel=time_label,
            output_path=output_dir / f"{args.prefix}_z.{args.format}",
        )
    )

    saved_paths.append(
        plot_component(
            estimated=estimated,
            ground_truth=ground_truth,
            time_origin=time_origin,
            estimated_values=estimated.velocities_camera[:, 0],
            gt_values=get_gt_values(ground_truth, "velocity", 0),
            title="Camera velocity vx(t)",
            ylabel="vx [m/s]",
            xlabel=time_label,
            output_path=output_dir / f"{args.prefix}_vx.{args.format}",
        )
    )

    saved_paths.append(
        plot_component(
            estimated=estimated,
            ground_truth=ground_truth,
            time_origin=time_origin,
            estimated_values=estimated.velocities_camera[:, 1],
            gt_values=get_gt_values(ground_truth, "velocity", 1),
            title="Camera velocity vy(t)",
            ylabel="vy [m/s]",
            xlabel=time_label,
            output_path=output_dir / f"{args.prefix}_vy.{args.format}",
        )
    )

    saved_paths.append(
        plot_component(
            estimated=estimated,
            ground_truth=ground_truth,
            time_origin=time_origin,
            estimated_values=estimated.velocities_camera[:, 2],
            gt_values=get_gt_values(ground_truth, "velocity", 2),
            title="Camera velocity vz(t)",
            ylabel="vz [m/s]",
            xlabel=time_label,
            output_path=output_dir / f"{args.prefix}_vz.{args.format}",
        )
    )

    saved_paths.append(
        plot_component(
            estimated=estimated,
            ground_truth=ground_truth,
            time_origin=time_origin,
            estimated_values=estimated.speeds,
            gt_values=get_gt_values(ground_truth, "speed", 0),
            title="Camera speed(t)",
            ylabel="speed [m/s]",
            xlabel=time_label,
            output_path=output_dir / f"{args.prefix}_speed.{args.format}",
        )
    )

    saved_paths.append(
        plot_trajectory_3d(
            estimated=estimated,
            ground_truth=ground_truth,
            output_path=output_dir / f"{args.prefix}_3d.{args.format}",
        )
    )

    print("Trajectory plots saved")
    print("=" * 80)

    for path in saved_paths:
        print(path)

    print()
    print(f"samples: {len(estimated.trajectory)}")
    print(f"time_start: {estimated.timestamps[0]:.9f}")
    print(f"time_end:   {estimated.timestamps[-1]:.9f}")
    print(
        "final_position: "
        f"[{estimated.positions[-1, 0]:.6f}, "
        f"{estimated.positions[-1, 1]:.6f}, "
        f"{estimated.positions[-1, 2]:.6f}]"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot trajectory and velocity from EvSLAM result files."
    )
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--gt", default=None, type=Path)
    parser.add_argument(
        "--output-dir",
        default=Path("outputs/trajectory_plots"),
        type=Path,
    )
    parser.add_argument("--prefix", default="trajectory")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    parser.add_argument("--absolute-time", action="store_true")
    return parser.parse_args()


def load_trajectory_plot_data(path: Path, label: str) -> TrajectoryPlotData:
    data = load_evslam_result_array(path)

    trajectory = Trajectory.from_tum_array(data[:, :8])
    velocities_camera = data[:, 8:11].astype(np.float64)

    return TrajectoryPlotData(
        label=label,
        trajectory=trajectory,
        velocities_camera=velocities_camera,
    )


def get_gt_values(
    ground_truth: TrajectoryPlotData | None,
    value_type: str,
    column: int,
):
    if ground_truth is None:
        return None

    if value_type == "position":
        return ground_truth.positions[:, column]

    if value_type == "velocity":
        return ground_truth.velocities_camera[:, column]

    if value_type == "speed":
        return ground_truth.speeds

    raise ValueError(f"Unknown value_type: {value_type}")


def plot_component(
    estimated: TrajectoryPlotData,
    ground_truth: TrajectoryPlotData | None,
    time_origin: float,
    estimated_values: np.ndarray,
    gt_values,
    title: str,
    ylabel: str,
    xlabel: str,
    output_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(9.0, 4.8))

    ax.plot(
        estimated.timestamps - time_origin,
        estimated_values,
        linewidth=1.5,
        label=estimated.label,
    )

    if ground_truth is not None and gt_values is not None:
        ax.plot(
            ground_truth.timestamps - time_origin,
            gt_values,
            linewidth=1.2,
            linestyle="--",
            label=ground_truth.label,
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.5, alpha=0.4)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_trajectory_3d(
    estimated: TrajectoryPlotData,
    ground_truth: TrajectoryPlotData | None,
    output_path: Path,
) -> Path:
    fig = plt.figure(figsize=(8.0, 7.0))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        estimated.positions[:, 0],
        estimated.positions[:, 1],
        estimated.positions[:, 2],
        linewidth=1.5,
        label=estimated.label,
    )

    if ground_truth is not None:
        ax.plot(
            ground_truth.positions[:, 0],
            ground_truth.positions[:, 1],
            ground_truth.positions[:, 2],
            linewidth=1.2,
            linestyle="--",
            label=ground_truth.label,
        )

    ax.scatter(
        estimated.positions[0, 0],
        estimated.positions[0, 1],
        estimated.positions[0, 2],
        marker="o",
        label="start",
    )

    ax.scatter(
        estimated.positions[-1, 0],
        estimated.positions[-1, 1],
        estimated.positions[-1, 2],
        marker="x",
        label="end",
    )

    ax.set_title("3D trajectory")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

    all_positions = estimated.positions
    if ground_truth is not None:
        all_positions = np.vstack((all_positions, ground_truth.positions))

    set_axes_equal_3d(ax, all_positions)

    ax.grid(True, linewidth=0.5, alpha=0.4)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return output_path


def set_axes_equal_3d(ax, points: np.ndarray) -> None:
    """
    Set equal scale for x, y and z axes in a 3D plot.
    """
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)

    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))

    if radius <= 0.0:
        radius = 1.0

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


if __name__ == "__main__":
    main()
