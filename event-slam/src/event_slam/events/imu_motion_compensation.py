from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from event_slam.core.camera import CameraModel
from event_slam.core.imu import imu_rotation_between_camera_times
from event_slam.core.types import EventBatch, StereoEventWindow


@dataclass
class MotionCompensationStats:
    """
    Basic diagnostics for one compensated event batch.
    """

    input_count: int
    output_count: int
    dropped_count: int
    reference_time: float


@dataclass
class StereoMotionCompensationResult:
    """
    Result of stereo event-window motion compensation.
    """

    window: StereoEventWindow
    left_stats: MotionCompensationStats
    right_stats: MotionCompensationStats


def compensate_stereo_window_rotation(
    window: StereoEventWindow,
    left_camera: CameraModel,
    right_camera: CameraModel,
    imu_timestamps: np.ndarray,
    angular_velocities: np.ndarray,
    T_C_left_imu: np.ndarray,
    T_C_right_imu: np.ndarray,
    timeshift_left_imu: float = 0.0,
    timeshift_right_imu: float = 0.0,
    imu_time_offset: float = 0.0,
    reference_time: str = "middle",
    gyro_bias: np.ndarray = None,
    num_time_bins: int = 32,
) -> StereoMotionCompensationResult:
    """
    Compensate rotational motion in a stereo event window.

    Only rotational compensation is applied. Translational compensation would
    require depth for each event, which is not available before stereo matching.
    """
    reference_camera_time = resolve_reference_time(
        t_start=window.t_start,
        t_end=window.t_end,
        reference_time=reference_time,
    )

    left_batch, left_stats = compensate_event_batch_rotation(
        batch=window.left,
        camera=left_camera,
        imu_timestamps=imu_timestamps,
        angular_velocities=angular_velocities,
        T_C_imu=T_C_left_imu,
        reference_camera_time=reference_camera_time,
        timeshift_cam_imu=timeshift_left_imu,
        imu_time_offset=imu_time_offset,
        gyro_bias=gyro_bias,
        num_time_bins=num_time_bins,
    )

    right_batch, right_stats = compensate_event_batch_rotation(
        batch=window.right,
        camera=right_camera,
        imu_timestamps=imu_timestamps,
        angular_velocities=angular_velocities,
        T_C_imu=T_C_right_imu,
        reference_camera_time=reference_camera_time,
        timeshift_cam_imu=timeshift_right_imu,
        imu_time_offset=imu_time_offset,
        gyro_bias=gyro_bias,
        num_time_bins=num_time_bins,
    )

    compensated_window = StereoEventWindow(
        t_start=window.t_start,
        t_end=window.t_end,
        left=left_batch,
        right=right_batch,
    )

    return StereoMotionCompensationResult(
        window=compensated_window,
        left_stats=left_stats,
        right_stats=right_stats,
    )


def compensate_event_batch_rotation(
    batch: EventBatch,
    camera: CameraModel,
    imu_timestamps: np.ndarray,
    angular_velocities: np.ndarray,
    T_C_imu: np.ndarray,
    reference_camera_time: float,
    timeshift_cam_imu: float = 0.0,
    imu_time_offset: float = 0.0,
    gyro_bias: np.ndarray = None,
    num_time_bins: int = 32,
) -> tuple:
    """
    Warp events to a reference time using gyro-only rotational compensation.

    For each event ray x(t), the compensated ray is:

        x_ref ~= R_Cref_Cevent @ x_event

    The implementation groups events into short time bins for speed. This keeps
    the operation vectorized while still approximating per-event compensation.
    """
    input_count = len(batch)

    if input_count == 0:
        stats = MotionCompensationStats(
            input_count=0,
            output_count=0,
            dropped_count=0,
            reference_time=float(reference_camera_time),
        )
        return batch, stats

    num_time_bins = max(1, int(num_time_bins))

    rays = _events_to_rays(
        x=batch.x,
        y=batch.y,
        camera=camera,
    )

    warped_x = np.empty(input_count, dtype=np.int32)
    warped_y = np.empty(input_count, dtype=np.int32)
    valid = np.zeros(input_count, dtype=bool)

    bin_boundaries = _make_time_bin_boundaries(batch.t, num_time_bins)
    for start, end in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        if start == end:
            continue

        event_time = float(np.mean(batch.t[start:end]))

        R_Cref_Cevent = imu_rotation_between_camera_times(
            imu_timestamps=imu_timestamps,
            angular_velocities=angular_velocities,
            camera_start_time=event_time,
            camera_end_time=reference_camera_time,
            T_C_imu=T_C_imu,
            timeshift_cam_imu=timeshift_cam_imu,
            imu_time_offset=imu_time_offset,
            gyro_bias=gyro_bias,
        )

        x_bin, y_bin, valid_bin = _warp_rays_to_pixels(
            rays=rays[start:end],
            R_Cref_Cevent=R_Cref_Cevent,
            camera=camera,
        )

        warped_x[start:end] = x_bin
        warped_y[start:end] = y_bin
        valid[start:end] = valid_bin

    output_batch = EventBatch(
        x=warped_x[valid],
        y=warped_y[valid],
        t=batch.t[valid],
        p=batch.p[valid],
        camera=batch.camera
    )

    output_count = len(output_batch)

    stats = MotionCompensationStats(
        input_count=input_count,
        output_count=output_count,
        dropped_count=input_count - output_count,
        reference_time=float(reference_camera_time),
    )

    return output_batch, stats


def resolve_reference_time(
    t_start: float,
    t_end: float,
    reference_time: str = "middle",
) -> float:
    """
    Resolve a named reference time inside an event window.
    """
    if reference_time == "start":
        return float(t_start)

    if reference_time == "middle":
        return 0.5 * (float(t_start) + float(t_end))

    if reference_time == "end":
        return float(t_end)

    raise ValueError(
        "reference_time must be one of: 'start', 'middle', 'end'. "
        f"Got: {reference_time}"
    )


def _events_to_rays(
    x: np.ndarray,
    y: np.ndarray,
    camera: CameraModel,
) -> np.ndarray:
    rays = np.empty((len(x), 3), dtype=np.float64)
    rays[:, 0] = x
    rays[:, 0] = (rays[:, 0] - camera.cx) / camera.fx
    rays[:, 1] = y
    rays[:, 1] = (rays[:, 1] - camera.cy) / camera.fy
    rays[:, 2] = 1.0
    return rays


def _warp_rays_to_pixels(
    rays: np.ndarray,
    R_Cref_Cevent: np.ndarray,
    camera: CameraModel,
) -> tuple:
    rays_ref = (R_Cref_Cevent @ rays.T).T

    z = rays_ref[:, 2]
    z_valid = z > 1e-9
    z_safe = np.where(z_valid, z, 1.0)

    u = camera.fx * (rays_ref[:, 0] / z_safe) + camera.cx
    v = camera.fy * (rays_ref[:, 1] / z_safe) + camera.cy

    x = np.rint(u).astype(np.int64)
    y = np.rint(v).astype(np.int64)

    valid = (
        z_valid
        & (x >= 0)
        & (x < int(camera.width))
        & (y >= 0)
        & (y < int(camera.height))
    )

    return x, y, valid


def _make_time_bin_boundaries(
    timestamps: np.ndarray,
    num_time_bins: int,
) -> np.ndarray:
    """Return slice boundaries for chronologically ordered event timestamps."""
    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)

    if len(timestamps) == 0:
        return np.zeros(2, dtype=np.int64)

    t_min = float(timestamps[0])
    t_max = float(timestamps[-1])

    if t_max <= t_min or num_time_bins <= 1:
        return np.array([0, len(timestamps)], dtype=np.int64)

    edges = np.linspace(t_min, t_max, num_time_bins + 1)[1:-1]
    boundaries = np.searchsorted(timestamps, edges, side="left")
    return np.concatenate(([0], boundaries, [len(timestamps)])).astype(np.int64)
