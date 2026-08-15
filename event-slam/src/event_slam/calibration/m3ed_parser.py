from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from event_slam.calibration.kalibr_parser import ImuCalibration
from event_slam.core.camera import CameraModel, StereoCalibration
from event_slam.core.geometry import invert_transform
from event_slam.datasets.m3ed_reader import HDF5_CACHE_BYTES, SUPPORTED_VERSION_PREFIX


def load_m3ed_stereo_calibration(path) -> StereoCalibration:
    """Load Prophesee stereo and OVC IMU extrinsics embedded in M3ED H5."""
    with _open_h5(path) as h5_file:
        _validate_version(h5_file)
        left = _load_camera(h5_file, "left")
        right = _load_camera(h5_file, "right")
        T_Pleft_Pleft = _load_transform(
            h5_file, "prophesee/left/calib/T_to_prophesee_left"
        )
        T_Pleft_Pright = _load_transform(
            h5_file, "prophesee/right/calib/T_to_prophesee_left"
        )
        T_Pleft_I = _load_transform(
            h5_file, "ovc/imu/calib/T_to_prophesee_left"
        )

    if left.resolution != right.resolution:
        raise ValueError(
            f"M3ED stereo resolutions differ: {left.resolution} and {right.resolution}"
        )
    if not np.allclose(T_Pleft_Pleft, np.eye(4), atol=1e-8):
        raise ValueError("Left Prophesee T_to_prophesee_left must be identity")

    T_Pright_Pleft = invert_transform(T_Pleft_Pright)
    return StereoCalibration(
        left=left,
        right=right,
        T_C_right_C_left=T_Pright_Pleft,
        T_C_left_imu=T_Pleft_I,
        T_C_right_imu=T_Pright_Pleft @ T_Pleft_I,
        timeshift_left_imu=0.0,
        timeshift_right_imu=0.0,
    )


def load_m3ed_imu_calibration(path) -> ImuCalibration:
    """Return the synchronized M3ED IMU metadata used by the current frontend."""
    with _open_h5(path) as h5_file:
        _validate_version(h5_file)
        _load_transform(h5_file, "ovc/imu/calib/T_to_prophesee_left")
    return ImuCalibration(topic="/ovc/imu", time_offset=0.0)


def _load_camera(h5_file, side: str) -> CameraModel:
    root = f"prophesee/{side}/calib"
    required = (
        "intrinsics",
        "distortion_coeffs",
        "resolution",
        "camera_model",
        "distortion_model",
    )
    missing = [f"/{root}/{name}" for name in required if f"{root}/{name}" not in h5_file]
    if missing:
        raise KeyError("Missing M3ED camera calibration paths: " + ", ".join(missing))

    intrinsics = np.asarray(h5_file[f"{root}/intrinsics"][()], dtype=np.float64).reshape(-1)
    distortion = np.asarray(
        h5_file[f"{root}/distortion_coeffs"][()], dtype=np.float64
    ).reshape(-1)
    resolution = np.asarray(h5_file[f"{root}/resolution"][()], dtype=np.int64).reshape(-1)
    camera_model = _read_text(h5_file[f"{root}/camera_model"][()])
    distortion_model = _read_text(h5_file[f"{root}/distortion_model"][()])

    if intrinsics.shape != (4,) or not np.all(np.isfinite(intrinsics)):
        raise ValueError(f"/{root}/intrinsics must contain four finite values")
    if intrinsics[0] <= 0 or intrinsics[1] <= 0:
        raise ValueError(f"/{root}/intrinsics focal lengths must be positive")
    if distortion.shape != (4,) or not np.all(np.isfinite(distortion)):
        raise ValueError(f"/{root}/distortion_coeffs must contain four finite values")
    if resolution.shape != (2,) or np.any(resolution <= 0):
        raise ValueError(f"/{root}/resolution must contain positive [width, height]")
    if camera_model != "pinhole" or distortion_model != "radtan":
        raise ValueError(
            f"Unsupported M3ED camera models: {camera_model}/{distortion_model}"
        )

    return CameraModel.from_intrinsics(
        name=side,
        width=int(resolution[0]),
        height=int(resolution[1]),
        fx=float(intrinsics[0]),
        fy=float(intrinsics[1]),
        cx=float(intrinsics[2]),
        cy=float(intrinsics[3]),
        distortion=distortion,
        camera_model=camera_model,
        distortion_model=distortion_model,
    )


def _load_transform(h5_file, path: str) -> np.ndarray:
    if path not in h5_file:
        raise KeyError(f"Missing required M3ED transform: /{path}")
    T_A_B = np.asarray(h5_file[path][()], dtype=np.float64)
    if T_A_B.shape != (4, 4) or not np.all(np.isfinite(T_A_B)):
        raise ValueError(f"/{path} must be a finite 4x4 transform")
    if not np.allclose(T_A_B[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(f"/{path} has an invalid homogeneous last row")
    R_A_B = T_A_B[:3, :3]
    if not np.allclose(R_A_B.T @ R_A_B, np.eye(3), atol=1e-6) or np.linalg.det(
        R_A_B
    ) <= 0.0:
        raise ValueError(f"/{path} has an invalid rotation")
    return T_A_B


def _open_h5(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"M3ED H5 file does not exist: {path}")
    return h5py.File(str(path), "r", rdcc_nbytes=HDF5_CACHE_BYTES)


def _validate_version(h5_file) -> None:
    version = _read_text(h5_file.attrs.get("version"))
    if not version.startswith(SUPPORTED_VERSION_PREFIX):
        raise ValueError(
            f"Unsupported M3ED version {version!r}; expected {SUPPORTED_VERSION_PREFIX}.x"
        )


def _read_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    raise ValueError(f"Expected an HDF5 string, got {type(value).__name__}")
