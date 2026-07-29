from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from event_slam.core.geometry import rotmat_to_quat_xyzw
from event_slam.core.trajectory import Trajectory
from event_slam.core.velocity import VelocityTrajectory, world_to_camera_velocities


@dataclass
class ResultWriterStats:
    """
    Statistics returned after writing a challenge result file.
    """

    reference_count: int
    written_count: int
    skipped_count: int
    first_written_timestamp: float = np.nan
    last_written_timestamp: float = np.nan


def load_reference_timestamps(path) -> np.ndarray:
    """
    Load reference timestamps from a text file.

    Empty lines and comment lines starting with '#' are ignored. If the file has
    multiple columns, the first column is interpreted as the timestamp.
    """
    path = Path(path)
    timestamps = []

    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            try:
                timestamps.append(float(parts[0]))
            except (IndexError, ValueError) as exc:
                raise ValueError(
                    f"Could not parse timestamp in {path} at line {line_number}: {line}"
                ) from exc

    return np.asarray(timestamps, dtype=np.float64)


def write_trajectory_at_timestamps(
    trajectory: Trajectory,
    velocity: VelocityTrajectory,
    timestamps: np.ndarray,
    output_path,
    skip_out_of_range: bool = True,
) -> ResultWriterStats:
    """
    Interpolate a trajectory at selected timestamps and save challenge output.

    Output format:
        timestamp tx ty tz qx qy qz qw vx vy vz

    The linear velocity is expressed in the current camera frame.
    """
    if trajectory.is_empty:
        raise ValueError("Cannot write result from an empty trajectory")

    if velocity.is_empty:
        raise ValueError("Cannot write result from an empty velocity trajectory")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)

    interpolated, skipped_count = _interpolate_valid_poses(
        trajectory=trajectory,
        timestamps=timestamps,
        skip_out_of_range=skip_out_of_range,
    )

    velocities_world = _interpolate_world_velocities(
        velocity=velocity,
        timestamps=interpolated.timestamps,
    )
    velocities_camera = world_to_camera_velocities(
        trajectory=interpolated,
        velocities_world=velocities_world,
    )

    written_count = 0
    first_written_timestamp = np.nan
    last_written_timestamp = np.nan

    with open(output_path, "w", encoding="utf-8") as file:
        for pose_sample, velocity_camera in zip(
            interpolated.samples,
            velocities_camera,
        ):
            timestamp = pose_sample.timestamp
            pose = pose_sample.pose

            tx, ty, tz = pose.t
            qx, qy, qz, qw = rotmat_to_quat_xyzw(pose.R)
            vx, vy, vz = velocity_camera

            file.write(
                f"{timestamp:.9f} "
                f"{tx:.9f} {ty:.9f} {tz:.9f} "
                f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f} "
                f"{vx:.9f} {vy:.9f} {vz:.9f}\n"
            )

            if written_count == 0:
                first_written_timestamp = float(timestamp)

            last_written_timestamp = float(timestamp)
            written_count += 1

    return ResultWriterStats(
        reference_count=len(timestamps),
        written_count=written_count,
        skipped_count=skipped_count,
        first_written_timestamp=first_written_timestamp,
        last_written_timestamp=last_written_timestamp,
    )


def write_result_from_reference_file(
    trajectory: Trajectory,
    velocity: VelocityTrajectory,
    reference_path,
    output_path,
    skip_out_of_range: bool = True,
) -> ResultWriterStats:
    """
    Load reference timestamps and write the trajectory interpolated to them.
    """
    timestamps = load_reference_timestamps(reference_path)

    return write_trajectory_at_timestamps(
        trajectory=trajectory,
        velocity=velocity,
        timestamps=timestamps,
        output_path=output_path,
        skip_out_of_range=skip_out_of_range,
    )


def _interpolate_valid_poses(
    trajectory: Trajectory,
    timestamps: np.ndarray,
    skip_out_of_range: bool,
) -> tuple:
    """
    Interpolate trajectory samples and optionally skip out-of-range timestamps.
    """
    output = Trajectory()
    skipped_count = 0

    for timestamp in timestamps:
        timestamp = float(timestamp)

        try:
            pose = trajectory.interpolate(timestamp, clamp=False)
        except ValueError as exc:
            if skip_out_of_range:
                skipped_count += 1
                continue

            raise ValueError(
                f"Reference timestamp {timestamp:.9f} is outside trajectory range"
            ) from exc

        output.append(timestamp=timestamp, pose=pose)

    return output, skipped_count


def _interpolate_world_velocities(
    velocity: VelocityTrajectory,
    timestamps: np.ndarray,
) -> np.ndarray:
    """
    Linearly interpolate world-frame velocity to selected timestamps.
    """
    source_timestamps = velocity.timestamps
    source_velocities = velocity.velocities_world
    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)

    if np.any(np.diff(source_timestamps) <= 0.0):
        raise ValueError("Velocity timestamps must be strictly increasing")

    if len(timestamps) == 0:
        return np.empty((0, 3), dtype=np.float64)

    if timestamps[0] < source_timestamps[0] or timestamps[-1] > source_timestamps[-1]:
        raise ValueError("Requested timestamps are outside velocity range")

    return np.column_stack(
        [
            np.interp(timestamps, source_timestamps, source_velocities[:, axis])
            for axis in range(3)
        ]
    ).astype(np.float64)
