from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml

from event_slam.core.camera import CameraModel, StereoCalibration
from event_slam.core.geometry import as_float_array


@dataclass
class ImuCalibration:
    """
    Relevant fields from a Kalibr-style IMU calibration.
    """

    topic: Optional[str] = None
    time_offset: Optional[float] = None
    T_imu_body: Optional[np.ndarray] = None
    gyroscope_noise_density: float = 1.7e-4
    gyroscope_random_walk: float = 2.0e-5
    accelerometer_noise_density: float = 2.0e-3
    accelerometer_random_walk: float = 3.0e-3


def load_stereo_calibration(camera_yaml_path) -> StereoCalibration:
    """
    Load a Kalibr-style camera YAML file and return a StereoCalibration object.

    Expected camera-chain convention:
    - cam0 is the left camera,
    - cam1 is the right camera,
    - cam1.T_cn_cnm1 is interpreted as T_C_right_C_left.

    T_C_right_C_left transforms points from the left camera frame to the right
    camera frame:

        p_C_right = T_C_right_C_left @ p_C_left
    """
    camera_yaml_path = Path(camera_yaml_path)
    data = _load_yaml(camera_yaml_path)

    cam0 = _require_mapping(data, "cam0", "camera YAML root")
    cam1 = _require_mapping(data, "cam1", "camera YAML root")

    left_camera = _parse_camera_model("cam0", "left", cam0)
    right_camera = _parse_camera_model("cam1", "right", cam1)

    T_C_left_imu = _parse_matrix4x4(
        _require_value(cam0, "T_cam_imu", "cam0"),
        "cam0.T_cam_imu",
    )

    T_C_right_imu = _parse_matrix4x4(
        _require_value(cam1, "T_cam_imu", "cam1"),
        "cam1.T_cam_imu",
    )

    T_C_right_C_left = _parse_matrix4x4(
        _require_value(cam1, "T_cn_cnm1", "cam1"),
        "cam1.T_cn_cnm1",
    )

    timeshift_left_imu = _parse_float(
        _require_value(cam0, "timeshift_cam_imu", "cam0"),
        "cam0.timeshift_cam_imu",
    )

    timeshift_right_imu = _parse_float(
        _require_value(cam1, "timeshift_cam_imu", "cam1"),
        "cam1.timeshift_cam_imu",
    )

    return StereoCalibration(
        left=left_camera,
        right=right_camera,
        T_C_right_C_left=T_C_right_C_left,
        T_C_left_imu=T_C_left_imu,
        T_C_right_imu=T_C_right_imu,
        timeshift_left_imu=timeshift_left_imu,
        timeshift_right_imu=timeshift_right_imu,
    )


def load_imu_calibration(imu_yaml_path) -> ImuCalibration:
    """
    Load the IMU fields used by the gyro-only frontend.
    """
    imu_yaml_path = Path(imu_yaml_path)
    data = _load_yaml(imu_yaml_path)

    imu_key = _find_first_key_with_prefix(data, "imu")
    imu_data = _require_mapping(data, imu_key, "IMU YAML root")

    time_offset = imu_data.get("time_offset")
    if time_offset is not None:
        time_offset = _parse_float(time_offset, "time_offset")

    T_imu_body = imu_data.get("T_i_b")
    if T_imu_body is not None:
        T_imu_body = _parse_matrix4x4(T_imu_body, "T_i_b")

    return ImuCalibration(
        topic=_optional_string(imu_data, "rostopic"),
        time_offset=time_offset,
        T_imu_body=T_imu_body,
        gyroscope_noise_density=float(
            imu_data.get("gyroscope_noise_density", 1.7e-4)
        ),
        gyroscope_random_walk=float(
            imu_data.get("gyroscope_random_walk", 2.0e-5)
        ),
        accelerometer_noise_density=float(
            imu_data.get("accelerometer_noise_density", 2.0e-3)
        ),
        accelerometer_random_walk=float(
            imu_data.get("accelerometer_random_walk", 3.0e-3)
        ),
    )


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"YAML file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping, got {type(data).__name__}: {path}")

    return data


def _parse_camera_model(camera_key: str, camera_name: str, camera_data: dict) -> CameraModel:
    intrinsics = _parse_intrinsics(
        _require_value(camera_data, "intrinsics", camera_key),
        f"{camera_key}.intrinsics",
    )

    distortion = _parse_distortion(
        _require_value(camera_data, "distortion_coeffs", camera_key),
        f"{camera_key}.distortion_coeffs",
    )

    width, height = _parse_resolution(
        _require_value(camera_data, "resolution", camera_key),
        f"{camera_key}.resolution",
    )

    camera_model = _parse_string(
        _require_value(camera_data, "camera_model", camera_key),
        f"{camera_key}.camera_model",
    )

    distortion_model = _parse_string(
        _require_value(camera_data, "distortion_model", camera_key),
        f"{camera_key}.distortion_model",
    )

    fx, fy, cx, cy = intrinsics

    return CameraModel.from_intrinsics(
        name=camera_name,
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        distortion=distortion,
        camera_model=camera_model,
        distortion_model=distortion_model,
    )


def _parse_intrinsics(value: object, field_name: str) -> Tuple[float, float, float, float]:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)

    if arr.shape != (4,):
        raise ValueError(
            f"{field_name} must contain [fx, fy, cx, cy], got shape {arr.shape}"
        )

    fx, fy, cx, cy = [float(v) for v in arr]

    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"{field_name} must contain positive focal lengths")

    return fx, fy, cx, cy


def _parse_distortion(value: object, field_name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)

    if arr.shape[0] not in (0, 4, 5, 8):
        raise ValueError(
            f"{field_name} must have 0, 4, 5 or 8 coefficients, got {arr.shape[0]}"
        )

    return arr


def _parse_resolution(value: object, field_name: str) -> Tuple[int, int]:
    arr = np.asarray(value, dtype=np.int64).reshape(-1)

    if arr.shape != (2,):
        raise ValueError(f"{field_name} must contain [width, height], got shape {arr.shape}")

    width, height = int(arr[0]), int(arr[1])

    if width <= 0 or height <= 0:
        raise ValueError(f"{field_name} must contain positive width and height")

    return width, height


def _parse_matrix4x4(value: object, field_name: str) -> np.ndarray:
    return as_float_array(value, (4, 4), field_name)


def _parse_float(value: object, field_name: str) -> float:
    try:
        return float(value)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be a float-like value") from exc
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a float-like value") from exc


def _parse_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string, got {type(value).__name__}")

    if not value:
        raise ValueError(f"{field_name} must not be empty")

    return value


def _require_value(mapping: dict, key: str, context: str) -> object:
    if key not in mapping:
        available = ", ".join(str(k) for k in mapping.keys())
        raise KeyError(f"Missing required key '{key}' in {context}. Available keys: {available}")

    return mapping[key]


def _require_mapping(mapping: dict, key: str, context: str) -> dict:
    value = _require_value(mapping, key, context)

    if not isinstance(value, dict):
        raise ValueError(
            f"Expected '{key}' in {context} to be a mapping, got {type(value).__name__}"
        )

    return value


def _optional_string(mapping: dict, key: str) -> Optional[str]:
    value = mapping.get(key)

    if value is None:
        return None

    if not isinstance(value, str):
        raise ValueError(f"Optional field '{key}' must be a string if present")

    return value


def _find_first_key_with_prefix(mapping: dict, prefix: str) -> str:
    matching_keys = sorted(str(key) for key in mapping.keys() if str(key).startswith(prefix))

    if not matching_keys:
        available = ", ".join(str(k) for k in mapping.keys())
        raise KeyError(
            f"Could not find any key starting with '{prefix}'. Available keys: {available}"
        )

    return matching_keys[0]
