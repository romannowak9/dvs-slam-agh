from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from event_slam.core.geometry import as_float_array
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
        self.velocity_world = as_float_array(
            self.velocity_world,
            (3,),
            "velocity_world",
        )
        self.velocity_camera = as_float_array(
            self.velocity_camera,
            (3,),
            "velocity_camera",
        )
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

def compute_velocity_trajectory(
    trajectory: Trajectory,
    smoothing_window_size: int = 1,
    smoothing_poly_order: int = 2,
) -> VelocityTrajectory:
    """
    Compute linear velocity from a pose trajectory.

    The input trajectory stores poses T_W_C. Therefore:
        - pose.t is the camera position expressed in the world frame,
        - pose.R maps vectors from camera frame C to world frame W.

    By default, velocity is estimated in the world frame with finite differences:
        v_W = dp_W / dt

    If smoothing_window_size > 1, velocity is estimated as the derivative of a
    local polynomial fitted to world-frame positions.

    Then it is expressed in the current camera frame:
        v_C = R_W_C.T @ v_W
    """
    velocity_world = compute_world_velocities(
        trajectory=trajectory,
        smoothing_window_size=smoothing_window_size,
        smoothing_poly_order=smoothing_poly_order,
    )
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
    smoothing_window_size: int = 1,
    smoothing_poly_order: int = 2,
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

    return compute_velocity_trajectory(
        trajectory=interpolated,
        smoothing_window_size=smoothing_window_size,
        smoothing_poly_order=smoothing_poly_order,
    )


def compute_world_velocities(
    trajectory: Trajectory,
    smoothing_window_size: int = 1,
    smoothing_poly_order: int = 2,
) -> np.ndarray:
    """
    Compute linear velocity in the world frame.

    By default, finite differences are used. If smoothing_window_size > 1,
    velocity is estimated as the derivative of a local polynomial fitted to
    world-frame positions.
    """
    if trajectory.is_empty:
        return np.empty((0, 3), dtype=np.float64)

    timestamps = trajectory.timestamps
    positions = trajectory.positions

    if smoothing_window_size <= 1:
        return finite_difference_velocity(
            timestamps=timestamps,
            positions=positions,
        )

    return local_polynomial_velocity(
        timestamps=timestamps,
        positions=positions,
        window_size=smoothing_window_size,
        poly_order=smoothing_poly_order,
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


def local_polynomial_velocity(
    timestamps: np.ndarray,
    positions: np.ndarray,
    window_size: int = 5,
    poly_order: int = 2,
) -> np.ndarray:
    """
    Estimate velocity from locally fitted position polynomials.

    For each timestamp, a polynomial is fitted to nearby world-frame positions:
        p(t) ~= a0 + a1 * t + a2 * t^2 + ...

    The velocity estimate is the first derivative at the current timestamp.
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

    if count < 3:
        return finite_difference_velocity(
            timestamps=timestamps,
            positions=positions,
        )

    window_size = int(window_size)
    poly_order = int(poly_order)

    if window_size % 2 == 0:
        raise ValueError(f"window_size must be odd, got {window_size}")

    if window_size < 3:
        raise ValueError(f"window_size must be at least 3, got {window_size}")

    if poly_order < 1:
        raise ValueError(f"poly_order must be at least 1, got {poly_order}")

    window_size = min(window_size, count if count % 2 == 1 else count - 1)

    if window_size < 3:
        return finite_difference_velocity(
            timestamps=timestamps,
            positions=positions,
        )

    poly_order = min(poly_order, window_size - 1)

    velocities = np.empty_like(positions)
    half_window = window_size // 2

    for index in range(count):
        start = max(0, min(index - half_window, count - window_size))
        end = start + window_size

        velocities[index] = _fit_local_polynomial_derivative(
            timestamps=timestamps[start:end],
            positions=positions[start:end],
            center_time=timestamps[index],
            poly_order=poly_order,
        )

    return velocities


def _fit_local_polynomial_derivative(
    timestamps: np.ndarray,
    positions: np.ndarray,
    center_time: float,
    poly_order: int,
) -> np.ndarray:
    local_t = timestamps - float(center_time)
    time_scale = float(np.max(np.abs(local_t)))

    if time_scale <= 1e-12:
        return np.zeros(3, dtype=np.float64)

    tau = local_t / time_scale
    A = np.vander(tau, N=poly_order + 1, increasing=True)

    coeffs, _, _, _ = np.linalg.lstsq(A, positions, rcond=None)

    return coeffs[1] / time_scale


def _position_delta_over_time(
    position1: np.ndarray,
    position0: np.ndarray,
    dt: float,
) -> np.ndarray:
    dt = float(dt)

    if dt <= 0.0:
        raise ValueError(f"Velocity requires strictly increasing timestamps, got dt={dt}")

    return (position1 - position0) / dt
