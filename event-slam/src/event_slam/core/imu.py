from __future__ import annotations

import numpy as np

from event_slam.core.geometry import (
    Pose,
    as_float_array,
    rotvec_to_rotmat,
)

from event_slam.datasets.evslam_reader import EvSlamRosbagReader


def camera_time_to_imu_time(
    camera_time: float,
    timeshift_cam_imu: float = 0.0,
    imu_time_offset: float = 0.0,
) -> float:
    """
    Convert a camera timestamp to the corresponding IMU timestamp.

    Kalibr convention:
        t_imu = t_cam + timeshift_cam_imu

    imu_time_offset is an additional optional offset from the IMU YAML file.
    For the current EvSLAM calibration it is 0.0.
    """
    return float(camera_time) + float(timeshift_cam_imu) + float(imu_time_offset)


def integrate_gyro_rotation(
    timestamps: np.ndarray,
    angular_velocities: np.ndarray,
    start_time: float,
    end_time: float,
    gyro_bias: np.ndarray = None,
) -> np.ndarray:
    """
    Integrate gyroscope measurements between two IMU timestamps.

    The returned rotation maps vectors from the previous IMU frame to the
    current IMU frame:

        X_Icurr = R_Icurr_Iprev @ X_Iprev

    Angular velocity is assumed to be expressed in the current IMU/body frame.
    A trapezoidal rule is used between available IMU samples.
    """
    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    angular_velocities = np.asarray(angular_velocities, dtype=np.float64)

    if angular_velocities.shape != (len(timestamps), 3):
        raise ValueError(
            "angular_velocities must have shape "
            f"({len(timestamps)}, 3), got {angular_velocities.shape}"
        )

    if len(timestamps) < 2:
        raise ValueError("At least two IMU samples are required")

    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("IMU timestamps must be strictly increasing")

    start_time = float(start_time)
    end_time = float(end_time)

    if end_time < start_time:
        return integrate_gyro_rotation(
            timestamps=timestamps,
            angular_velocities=angular_velocities,
            start_time=end_time,
            end_time=start_time,
            gyro_bias=gyro_bias,
        ).T

    if start_time == end_time:
        return np.eye(3, dtype=np.float64)

    if start_time < timestamps[0] or end_time > timestamps[-1]:
        raise ValueError(
            "Requested IMU integration interval is outside available IMU range: "
            f"[{start_time:.9f}, {end_time:.9f}] not within "
            f"[{timestamps[0]:.9f}, {timestamps[-1]:.9f}]"
        )

    if gyro_bias is None:
        gyro_bias = np.zeros(3, dtype=np.float64)
    else:
        gyro_bias = as_float_array(gyro_bias, (3,), "gyro_bias")

    segment_times = _integration_times(
        timestamps=timestamps,
        start_time=start_time,
        end_time=end_time,
    )
    segment_gyro = _interpolate_vectors(
        source_timestamps=timestamps,
        source_values=angular_velocities,
        target_timestamps=segment_times,
    )

    R_Iprev_Icurr = np.eye(3, dtype=np.float64)

    for index in range(len(segment_times) - 1):
        dt = segment_times[index + 1] - segment_times[index]
        omega = 0.5 * (segment_gyro[index] + segment_gyro[index + 1]) - gyro_bias
        R_Iprev_Icurr = R_Iprev_Icurr @ rotvec_to_rotmat(omega * dt)

    return R_Iprev_Icurr.T


def camera_rotation_from_imu_rotation(
    R_Icurr_Iprev: np.ndarray,
    T_C_imu: np.ndarray,
) -> np.ndarray:
    """
    Convert relative IMU rotation to the camera frame.

    Inputs:
        R_Icurr_Iprev:
            Rotation mapping previous IMU coordinates to current IMU coordinates.

        T_C_imu:
            Transform mapping IMU frame to camera frame.

    Output:
        R_Ccurr_Cprev:
            Rotation mapping previous camera coordinates to current camera
            coordinates, directly comparable with PnP relative rotation.
    """
    R_Icurr_Iprev = as_float_array(
        R_Icurr_Iprev,
        (3, 3),
        "R_Icurr_Iprev",
    )
    T_C_imu = as_float_array(T_C_imu, (4, 4), "T_C_imu")

    R_C_imu = T_C_imu[:3, :3]

    return R_C_imu @ R_Icurr_Iprev @ R_C_imu.T


def apply_camera_frame_correction(
    R_camera: np.ndarray,
    R_output_from_camera: np.ndarray = None,
) -> np.ndarray:
    """
    Express a camera-frame relative rotation in the configured output frame.

    If:
        X_output = C @ X_camera

    then:
        R_output = C @ R_camera @ C.T
    """
    R_camera = as_float_array(R_camera, (3, 3), "R_camera")

    if R_output_from_camera is None:
        return R_camera

    C = as_float_array(
        R_output_from_camera,
        (3, 3),
        "R_output_from_camera",
    )

    return C @ R_camera @ C.T


def imu_rotation_between_camera_times(
    imu_timestamps: np.ndarray,
    angular_velocities: np.ndarray,
    camera_start_time: float,
    camera_end_time: float,
    T_C_imu: np.ndarray,
    timeshift_cam_imu: float = 0.0,
    imu_time_offset: float = 0.0,
    gyro_bias: np.ndarray = None,
    R_output_from_camera: np.ndarray = None,
) -> np.ndarray:
    """
    Integrate gyro between two camera timestamps and return camera-frame rotation.

    The output rotation is directly comparable with a PnP relative rotation:
        X_Ccurr = R_Ccurr_Cprev @ X_Cprev

    If R_output_from_camera is provided, the result is additionally expressed
    in that output camera frame.
    """
    imu_start_time = camera_time_to_imu_time(
        camera_time=camera_start_time,
        timeshift_cam_imu=timeshift_cam_imu,
        imu_time_offset=imu_time_offset,
    )
    imu_end_time = camera_time_to_imu_time(
        camera_time=camera_end_time,
        timeshift_cam_imu=timeshift_cam_imu,
        imu_time_offset=imu_time_offset,
    )

    R_Icurr_Iprev = integrate_gyro_rotation(
        timestamps=imu_timestamps,
        angular_velocities=angular_velocities,
        start_time=imu_start_time,
        end_time=imu_end_time,
        gyro_bias=gyro_bias,
    )
    R_Ccurr_Cprev = camera_rotation_from_imu_rotation(
        R_Icurr_Iprev=R_Icurr_Iprev,
        T_C_imu=T_C_imu,
    )

    return apply_camera_frame_correction(
        R_camera=R_Ccurr_Cprev,
        R_output_from_camera=R_output_from_camera,
    )


def relative_camera_rotation_from_poses(
    previous_pose: Pose,
    current_pose: Pose,
) -> np.ndarray:
    """
    Compute relative camera rotation from two poses T_W_C.

    Pose.R maps camera vectors to world vectors. Therefore:

        R_Ccurr_Cprev = R_W_Ccurr.T @ R_W_Cprev
    """
    return current_pose.R.T @ previous_pose.R


def rotation_angle_deg(R: np.ndarray) -> float:
    """
    Return the rotation angle of a rotation matrix in degrees.
    """
    R = as_float_array(R, (3, 3), "R")

    cos_angle = 0.5 * (np.trace(R) - 1.0)
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))

    return float(np.degrees(np.arccos(cos_angle)))


def rotation_error_deg(R_estimated: np.ndarray, R_reference: np.ndarray) -> float:
    """
    Return the angular difference between two relative rotations.
    """
    R_estimated = as_float_array(R_estimated, (3, 3), "R_estimated")
    R_reference = as_float_array(R_reference, (3, 3), "R_reference")

    return rotation_angle_deg(R_estimated @ R_reference.T)


def extract_imu_sample(sample) -> tuple:
    """
    Extract timestamp and angular velocity from an IMU sample object.
    """
    timestamp = _read_timestamp(sample)
    angular_velocity = _read_vector3_field(
        sample=sample,
        field_names=("angular_velocity", "gyro", "gyroscope"),
    )

    return timestamp, angular_velocity

def prepare_imu_gyro_samples(
    timestamps: np.ndarray,
    angular_velocities: np.ndarray,
) -> tuple:
    """
    Sort IMU samples by timestamp and remove repeated timestamps.
    """
    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    angular_velocities = np.asarray(angular_velocities, dtype=np.float64)

    if angular_velocities.shape != (len(timestamps), 3):
        raise ValueError(
            "angular_velocities must have shape "
            f"({len(timestamps)}, 3), got {angular_velocities.shape}"
        )

    order = np.argsort(timestamps, kind="mergesort")
    timestamps = timestamps[order]
    angular_velocities = angular_velocities[order]

    keep = np.ones(len(timestamps), dtype=bool)
    keep[1:] = np.diff(timestamps) > 0.0

    return timestamps[keep], angular_velocities[keep]


def load_imu_gyro_from_evslam_reader(
    reader: EvSlamRosbagReader,
    imu_topic: str,
) -> tuple:
    timestamps = []
    angular_velocities = []

    for sample in reader.iter_imu_samples(topic=imu_topic):
        timestamp, angular_velocity = extract_imu_sample(sample)
        timestamps.append(timestamp)
        angular_velocities.append(angular_velocity)

    if len(timestamps) < 2:
        raise ValueError(
            f"Need at least two IMU samples on topic {imu_topic}, "
            f"got {len(timestamps)}"
        )

    return prepare_imu_gyro_samples(
        timestamps=np.asarray(timestamps, dtype=np.float64),
        angular_velocities=np.asarray(angular_velocities, dtype=np.float64),
    )


def _integration_times(
    timestamps: np.ndarray,
    start_time: float,
    end_time: float,
) -> np.ndarray:
    inside_mask = (timestamps > start_time) & (timestamps < end_time)

    return np.concatenate(
        (
            np.array([start_time], dtype=np.float64),
            timestamps[inside_mask],
            np.array([end_time], dtype=np.float64),
        )
    )


def _interpolate_vectors(
    source_timestamps: np.ndarray,
    source_values: np.ndarray,
    target_timestamps: np.ndarray,
) -> np.ndarray:
    output = np.empty((len(target_timestamps), source_values.shape[1]), dtype=np.float64)

    for axis in range(source_values.shape[1]):
        output[:, axis] = np.interp(
            target_timestamps,
            source_timestamps,
            source_values[:, axis],
        )

    return output


def _read_timestamp(sample) -> float:
    for field_name in ("timestamp", "t", "time"):
        if hasattr(sample, field_name):
            return float(getattr(sample, field_name))

    raise ValueError(f"Could not read timestamp from IMU sample: {sample}")


def _read_vector3_field(sample, field_names) -> np.ndarray:
    for field_name in field_names:
        if hasattr(sample, field_name):
            value = getattr(sample, field_name)

            if all(hasattr(value, attr) for attr in ("x", "y", "z")):
                vector = np.array([value.x, value.y, value.z], dtype=np.float64)
            else:
                vector = np.asarray(value, dtype=np.float64).reshape(-1)

            return as_float_array(vector, (3,), field_name)

    raise ValueError(
        f"Could not find any of {field_names} in IMU sample: {sample}"
    )