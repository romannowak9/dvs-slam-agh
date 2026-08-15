#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"

if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))


from event_slam.debug.slam_plots import (
    save_sparse_map_plot,
    save_tracking_diagnostics_plot,
)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.results_dir / "slam_plots"

    trajectory = _read_csv(args.results_dir / "trajectory.csv")
    keyframes = _read_csv(args.results_dir / "keyframes.csv")
    landmarks = _read_csv(args.results_dir / "landmarks.csv")

    map_path = save_sparse_map_plot(
        keyframe_positions_W=_matrix(keyframes, "tx", "ty", "tz"),
        landmark_positions_W=_matrix(landmarks, "x_world", "y_world", "z_world"),
        landmark_positions_C_anchor=_matrix(
            landmarks, "x_anchor", "y_anchor", "z_anchor"
        ),
        observation_counts=_values(landmarks, "observation_count", int),
        path=output_dir / "map_3d.png",
        max_plot_distance=args.max_plot_distance,
    )
    tracking_path = save_tracking_diagnostics_plot(
        track_counts=_values(trajectory, "track_count", int),
        new_feature_counts=_values(trajectory, "new_feature_count", int),
        map_point_counts=_values(trajectory, "map_point_count", int),
        map_inlier_counts=_values(trajectory, "map_inlier_count", int),
        pnp_inlier_counts=_values(trajectory, "pnp_inlier_count", int),
        pose_sources=_values(trajectory, "pose_source", str),
        descriptor_match_counts=_values(
            trajectory, "map_descriptor_match_count", int
        ),
        path=output_dir / "tracking_diagnostics.png",
    )

    print(f"frames: {len(trajectory)}")
    print(f"keyframes: {len(keyframes)}")
    print(f"landmarks: {len(landmarks)}")
    print(f"map_plot: {map_path}")
    print(f"tracking_plot: {tracking_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create SLAM map plots from previously saved CSV results."
    )
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-plot-distance", default=5.0, type=float)
    return parser.parse_args()


def _read_csv(path: Path) -> list:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"CSV contains no results: {path}")
    return rows


def _values(rows: list, name: str, value_type) -> np.ndarray:
    try:
        return np.asarray([value_type(row[name]) for row in rows])
    except KeyError:
        raise ValueError(f"CSV does not contain required column: {name}")


def _matrix(rows: list, *names: str) -> np.ndarray:
    return np.asarray(
        [[float(row[name]) for name in names] for row in rows],
        dtype=np.float64,
    )


if __name__ == "__main__":
    main()
