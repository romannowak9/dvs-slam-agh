#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"

if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))

from event_slam.calibration.kalibr_parser import (
    load_imu_calibration,
    load_stereo_calibration,
)
from event_slam.core.imu import (
    ImuCoverageError,
    camera_time_to_imu_time,
    imu_rotation_between_camera_times,
    relative_camera_rotation_from_poses,
    rotation_angle_deg,
)
from event_slam.core.trajectory import Trajectory
from event_slam.io.result_io import load_evslam_result
from event_slam.datasets.evslam_reader import EvSlamRosbagReader
from verification_config import load_args, verification_parser


R_OUT_FROM_PNP_CAMERA = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)


def main() -> None:
    _, args = parse_args()

    stereo = load_stereo_calibration(args.camera_calibration)
    imu_calibration = load_imu_calibration(args.imu_calibration)

    imu_topic = args.imu_topic or imu_calibration.topic

    if imu_topic is None:
        raise ValueError(
            "IMU topic was not provided and could not be read from IMU calibration"
        )

    trajectory, _ = load_evslam_result(args.estimate)
    imu_timestamps, angular_velocities = EvSlamRosbagReader(
        bag_path=args.bag
    ).load_imu_gyro(topic=imu_topic)

    timeshift_cam_imu = stereo.timeshift_left_imu + args.extra_timeshift
    imu_time_offset = imu_calibration.time_offset or 0.0

    rows = compute_rotation_diagnostics(
        trajectory=trajectory,
        imu_timestamps=imu_timestamps,
        angular_velocities=angular_velocities,
        T_C_imu=stereo.T_C_left_imu,
        timeshift_cam_imu=timeshift_cam_imu,
        imu_time_offset=imu_time_offset,
        max_pairs=args.max_pairs,
    )

    print_report(
        rows=rows,
        bag_path=args.bag,
        estimate_path=args.estimate,
        imu_topic=imu_topic,
        timeshift_cam_imu=timeshift_cam_imu,
        imu_time_offset=imu_time_offset,
        preview_count=args.preview_count,
    )

    if args.csv is not None:
        save_csv(args.csv, rows)


def compute_rotation_diagnostics(
    trajectory: Trajectory,
    imu_timestamps: np.ndarray,
    angular_velocities: np.ndarray,
    T_C_imu: np.ndarray,
    timeshift_cam_imu: float,
    imu_time_offset: float,
    max_pairs: int = 0,
) -> np.ndarray:
    """
    Compute PnP-vs-IMU relative rotation diagnostics for consecutive pose pairs.
    """
    rows = []
    pair_count = max(0, len(trajectory) - 1)

    if max_pairs > 0:
        pair_count = min(pair_count, int(max_pairs))

    for index in range(pair_count):
        previous_sample = trajectory.samples[index]
        current_sample = trajectory.samples[index + 1]

        previous_camera_time = previous_sample.timestamp
        current_camera_time = current_sample.timestamp

        previous_imu_time = camera_time_to_imu_time(
            camera_time=previous_camera_time,
            timeshift_cam_imu=timeshift_cam_imu,
            imu_time_offset=imu_time_offset,
        )
        current_imu_time = camera_time_to_imu_time(
            camera_time=current_camera_time,
            timeshift_cam_imu=timeshift_cam_imu,
            imu_time_offset=imu_time_offset,
        )

        try:
            R_imu = imu_rotation_between_camera_times(
                imu_timestamps=imu_timestamps,
                angular_velocities=angular_velocities,
                camera_start_time=previous_camera_time,
                camera_end_time=current_camera_time,
                T_C_imu=T_C_imu,
                timeshift_cam_imu=timeshift_cam_imu,
                imu_time_offset=imu_time_offset,
                R_output_from_camera=R_OUT_FROM_PNP_CAMERA,
            )
        except ImuCoverageError:
            continue

        R_pnp = relative_camera_rotation_from_poses(
            previous_pose=previous_sample.pose,
            current_pose=current_sample.pose,
        )
        R_error = R_pnp @ R_imu.T

        rows.append(
            [
                previous_camera_time,
                current_camera_time,
                previous_imu_time,
                current_imu_time,
                rotation_angle_deg(R_pnp),
                rotation_angle_deg(R_imu),
                rotation_angle_deg(R_error),
            ]
        )

    if not rows:
        raise ValueError("No valid PnP-vs-IMU rotation pairs were computed")

    return np.asarray(rows, dtype=np.float64)


def print_report(
    rows: np.ndarray,
    bag_path: Path,
    estimate_path: Path,
    imu_topic: str,
    timeshift_cam_imu: float,
    imu_time_offset: float,
    preview_count: int,
) -> None:
    errors = rows[:, 6]

    print("IMU rotation debug")
    print("=" * 80)
    print(f"bag:                 {bag_path}")
    print(f"estimate:            {estimate_path}")
    print(f"imu_topic:           {imu_topic}")
    print(f"timeshift_cam_imu:   {timeshift_cam_imu:.9f} s")
    print(f"imu_time_offset:     {imu_time_offset:.9f} s")
    print(f"compared pairs:      {len(rows)}")
    print()
    print("PnP-vs-IMU rotation error [deg]")
    print(f"mean:                {np.mean(errors):.6f}")
    print(f"median:              {np.median(errors):.6f}")
    print(f"max:                 {np.max(errors):.6f}")
    print()

    print("First rows:")
    print(
        "timestamp_prev timestamp_curr "
        "pnp_rotation_deg imu_rotation_deg pnp_imu_error_deg"
    )

    for row in rows[: int(preview_count)]:
        print(
            f"{row[0]:.9f} {row[1]:.9f} "
            f"{row[4]:.6f} {row[5]:.6f} {row[6]:.6f}"
        )


def save_csv(path: Path, rows: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savetxt(
        path,
        rows,
        fmt="%.9f",
        delimiter=",",
        header=(
            "timestamp_prev,timestamp_curr,"
            "imu_time_prev,imu_time_curr,"
            "pnp_rotation_deg,imu_rotation_deg,pnp_imu_error_deg"
        ),
        comments="",
    )


def parse_args() -> tuple:
    parser = verification_parser(
        "Compare relative rotations estimated from poses and IMU."
    )
    parser.add_argument("--estimate", type=Path, required=True)
    parser.add_argument("--extra-timeshift", default=0.0, type=float)
    parser.add_argument("--max-pairs", default=0, type=int)
    parser.add_argument("--preview-count", default=8, type=int)
    parser.add_argument("--csv", default=None, type=Path)
    return load_args(parser)


if __name__ == "__main__":
    main()
