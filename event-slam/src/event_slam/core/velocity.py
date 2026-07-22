from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from event_slam.core.trajectory import Trajectory


@dataclass
class VelocitySample:
    """
    A single linear velocity sample.

    velocity_world:
        Linear velocity expressed in the world frame W.

    velocity_camera:
        Linear velocity expressed in the current camera frame C.
    """

    timestamp: float
    velocity_world: np.ndarray
    velocity_camera: np.ndarray
    speed: float = 0.0

    def __post_init__(self) -> None:
        self.timestamp = float(self.timestamp)
        self.velocity_world = _as_vector3(self.velocity_world)
        self.velocity_camera = _as_vector3(self.velocity_camera)
        self.speed = float(np.linalg.norm(self.velocity_camera))


@dataclass
class VelocityTrajectory:
    """
    Linear velocity trajectory associated with a pose trajectory.
    """

    samples: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def is_empty(self) -> bool:
        return len(self.samples) == 0

    @property
    def timestamps(self) -> np.ndarray:
        return np.array([sample.timestamp for sample in self.samples], dtype=np.float64)

    @property
    def velocities_world(self) -> np.ndarray:
        if self.is_empty:
            return np.empty((0, 3), dtype=np.float64)

        return np.vstack([sample.velocity_world for sample in self.samples]).astype(
            np.float64
        )

    @property
    def velocities_camera(self) -> np.ndarray:
        if self.is_empty:
            return np.empty((0, 3), dtype=np.float64)

        return np.vstack([sample.velocity_camera for sample in self.samples]).astype(
            np.float64
        )

    @property
    def speeds(self) -> np.ndarray:
        return np.array([sample.speed for sample in self.samples], dtype=np.float64)

    def append(
        self,
        timestamp: float,
        velocity_world: np.ndarray,
        velocity_camera: np.ndarray,
    ) -> None:
        timestamp = float(timestamp)

        if self.samples and timestamp < self.samples[-1].timestamp:
            raise ValueError(
                "Velocity timestamps must be non-decreasing. "
                f"Last={self.samples[-1].timestamp}, new={timestamp}"
            )

        self.samples.append(
            VelocitySample(
                timestamp=timestamp,
                velocity_world=velocity_world,
                velocity_camera=velocity_camera,
            )
        )

    def as_camera_array(self) -> np.ndarray:
        """
        Return velocity in camera frame:

            timestamp vx vy vz speed
        """
        if self.is_empty:
            return np.empty((0, 5), dtype=np.float64)

        return np.hstack(
            (
                self.timestamps.reshape(-1, 1),
                self.velocities_camera,
                self.speeds.reshape(-1, 1),
            )
        ).astype(np.float64)

    def as_world_array(self) -> np.ndarray:
        """
        Return velocity in world frame:

            timestamp vx vy vz speed
        """
        if self.is_empty:
            return np.empty((0, 5), dtype=np.float64)

        speed = np.linalg.norm(self.velocities_world, axis=1)

        return np.hstack(
            (
                self.timestamps.reshape(-1, 1),
                self.velocities_world,
                speed.reshape(-1, 1),
            )
        ).astype(np.float64)

    def save_csv(self, path, camera_frame: bool = True) -> None:
        """
        Save velocity samples to a CSV file.

        By default, velocity is saved in the camera frame because this is the
        format needed by the EvSLAM evaluation output.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if camera_frame:
            data = self.as_camera_array()
            header = "timestamp,vx_camera,vy_camera,vz_camera,speed"
        else:
            data = self.as_world_array()
            header = "timestamp,vx_world,vy_world,vz_world,speed"

        np.savetxt(
            path,
            data,
            fmt="%.9f",
            delimiter=",",
            header=header,
            comments="",
        )


def compute_velocity_trajectory(trajectory: Trajectory) -> VelocityTrajectory:
    """
    Compute linear velocity from a pose trajectory.

    The input trajectory stores poses T_W_C. Therefore:
        - pose.t is the camera position expressed in the world frame,
        - pose.R maps vectors from camera frame C to world frame W.

    Velocity is first estimated in the world frame with finite differences:
        v_W = dp_W / dt

    Then it is expressed in the current camera frame:
        v_C = R_W_C.T @ v_W
    """
    velocity_world = compute_world_velocities(trajectory)
    velocity_camera = world_to_camera_velocities(trajectory, velocity_world)

    output = VelocityTrajectory()

    for timestamp, v_w, v_c in zip(
        trajectory.timestamps,
        velocity_world,
        velocity_camera,
    ):
        output.append(
            timestamp=float(timestamp),
            velocity_world=v_w,
            velocity_camera=v_c,
        )

    return output


def compute_velocity_at_timestamps(
    trajectory: Trajectory,
    timestamps: np.ndarray,
    clamp: bool = False,
) -> VelocityTrajectory:
    """
    Interpolate trajectory to selected timestamps and compute velocity there.

    This is useful for challenge output files, where poses and velocities must
    match reference timestamps one-to-one.
    """
    interpolated = trajectory.interpolate_many(
        np.asarray(timestamps, dtype=np.float64),
        clamp=clamp,
    )

    return compute_velocity_trajectory(interpolated)


def compute_world_velocities(trajectory: Trajectory) -> np.ndarray:
    """
    Compute linear velocity in the world frame using finite differences.

    Central differences are used for inner samples. Forward/backward differences
    are used for the first and last sample.
    """
    if trajectory.is_empty:
        return np.empty((0, 3), dtype=np.float64)

    timestamps = trajectory.timestamps
    positions = trajectory.positions

    return finite_difference_velocity(
        timestamps=timestamps,
        positions=positions,
    )


def world_to_camera_velocities(
    trajectory: Trajectory,
    velocities_world: np.ndarray,
) -> np.ndarray:
    """
    Express world-frame linear velocities in the corresponding camera frames.

    For a pose T_W_C, the rotation R_W_C maps vectors from C to W.
    Therefore the inverse rotation maps world vectors to camera vectors:
        v_C = R_W_C.T @ v_W
    """
    velocities_world = np.asarray(velocities_world, dtype=np.float64)

    if velocities_world.shape != (len(trajectory), 3):
        raise ValueError(
            "velocities_world must have shape "
            f"({len(trajectory)}, 3), got {velocities_world.shape}"
        )

    velocities_camera = np.empty_like(velocities_world)

    for index, sample in enumerate(trajectory.samples):
        R_W_C = sample.pose.R
        velocities_camera[index] = R_W_C.T @ velocities_world[index]

    return velocities_camera


def finite_difference_velocity(
    timestamps: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    """
    Estimate linear velocity from sampled positions.

    For N >= 3:
        - first sample: forward difference,
        - inner samples: central difference,
        - last sample: backward difference.

    For N == 1, zero velocity is returned.
    """
    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    positions = np.asarray(positions, dtype=np.float64)

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {positions.shape}")

    if len(timestamps) != len(positions):
        raise ValueError(
            "timestamps and positions must have the same length. "
            f"Got {len(timestamps)} and {len(positions)}"
        )

    count = len(timestamps)

    if count == 0:
        return np.empty((0, 3), dtype=np.float64)

    if count == 1:
        return np.zeros((1, 3), dtype=np.float64)

    velocities = np.empty((count, 3), dtype=np.float64)

    velocities[0] = _position_delta_over_time(
        positions[1],
        positions[0],
        timestamps[1] - timestamps[0],
    )

    for index in range(1, count - 1):
        velocities[index] = _position_delta_over_time(
            positions[index + 1],
            positions[index - 1],
            timestamps[index + 1] - timestamps[index - 1],
        )

    velocities[-1] = _position_delta_over_time(
        positions[-1],
        positions[-2],
        timestamps[-1] - timestamps[-2],
    )

    return velocities


def _position_delta_over_time(
    position1: np.ndarray,
    position0: np.ndarray,
    dt: float,
) -> np.ndarray:
    dt = float(dt)

    if dt <= 0.0:
        raise ValueError(f"Velocity requires strictly increasing timestamps, got dt={dt}")

    return (position1 - position0) / dt


def _as_vector3(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)

    if vector.shape != (3,):
        raise ValueError(f"Expected a 3D vector, got shape {vector.shape}")

    return vector