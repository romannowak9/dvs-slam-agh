#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"

if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))


from event_slam.calibration.kalibr_parser import (
    load_imu_calibration,
    load_stereo_calibration,
)
from event_slam.debug.visualization import print_section, print_vector


def main() -> None:
    args = parse_args()

    stereo = load_stereo_calibration(args.camera_yaml)
    imu = load_imu_calibration(args.imu_yaml) if args.imu_yaml is not None else None

    print_section("Stereo calibration summary")
    print(stereo.summary())

    print_section("Left camera")
    print_camera(stereo.left)

    print_section("Right camera")
    print_camera(stereo.right)

    print_section("Stereo extrinsics")
    print_matrix("T_C_right_C_left", stereo.T_C_right_C_left)
    print_matrix("T_C_left_C_right", stereo.T_C_left_C_right)
    print(f"Baseline [m]: {stereo.baseline:.9f}")

    print_section("Camera-IMU extrinsics")
    print_matrix_or_none("T_C_left_imu", stereo.T_C_left_imu)
    print_matrix_or_none("T_C_right_imu", stereo.T_C_right_imu)

    print_section("Camera-IMU time shifts")
    print(f"timeshift_left_imu  [s]: {stereo.timeshift_left_imu:.12f}")
    print(f"timeshift_right_imu [s]: {stereo.timeshift_right_imu:.12f}")

    print_section("IMU calibration")
    if imu is None:
        print("No IMU YAML file provided.")
    else:
        print(f"topic: {imu.topic}")
        print(f"time_offset: {imu.time_offset}")
        print_matrix_or_none("T_imu_body", imu.T_imu_body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug EvSLAM/Kalibr camera and IMU calibration files."
    )

    parser.add_argument(
        "--camera-yaml",
        required=True,
        type=Path,
        help="Path to calib_results_cam_drone.yaml.",
    )

    parser.add_argument(
        "--imu-yaml",
        default=None,
        type=Path,
        help="Optional path to calib_results_imu_drone.yaml.",
    )

    return parser.parse_args()


def print_camera(camera) -> None:
    print(f"name: {camera.name}")
    print(f"resolution: {camera.width} x {camera.height}")
    print(f"camera_model: {camera.camera_model}")
    print(f"distortion_model: {camera.distortion_model}")
    print_matrix("K", camera.K)
    print_vector("D", camera.D)


def print_matrix(name: str, matrix: np.ndarray) -> None:
    print(f"{name}:")
    print(np.array2string(matrix, precision=9, suppress_small=False))


def print_matrix_or_none(name: str, matrix: np.ndarray | None) -> None:
    if matrix is None:
        print(f"{name}: None")
        return

    print_matrix(name, matrix)


if __name__ == "__main__":
    main()