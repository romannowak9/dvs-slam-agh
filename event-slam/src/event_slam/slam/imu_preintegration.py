from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from event_slam.core.geometry import rotvec_to_rotmat, skew


@dataclass
class ImuPreintegration:
    """Bias-linearized IMU motion between two keyframes."""

    dt: float
    delta_R: np.ndarray
    delta_v: np.ndarray
    delta_p: np.ndarray
    covariance: np.ndarray
    J_R_bg: np.ndarray
    J_v_bg: np.ndarray
    J_v_ba: np.ndarray
    J_p_bg: np.ndarray
    J_p_ba: np.ndarray
    gyro_bias: np.ndarray
    accel_bias: np.ndarray
    sample_count: int

def preintegrate_imu(
    imu_data,
    start_time: float,
    end_time: float,
    gyro_bias: np.ndarray,
    accel_bias: np.ndarray,
    gyro_noise_density: float,
    accel_noise_density: float,
    gyro_random_walk: float,
    accel_random_walk: float,
) -> ImuPreintegration:
    """Preintegrate body-frame IMU samples in the starting IMU frame."""
    times, gyro, accel = imu_data.interval(float(start_time), float(end_time))
    gyro_bias = np.asarray(gyro_bias, dtype=np.float64).reshape(3)
    accel_bias = np.asarray(accel_bias, dtype=np.float64).reshape(3)
    delta_R = np.eye(3, dtype=np.float64)
    delta_v = np.zeros(3, dtype=np.float64)
    delta_p = np.zeros(3, dtype=np.float64)
    covariance = np.zeros((15, 15), dtype=np.float64)
    J_R_bg = np.zeros((3, 3), dtype=np.float64)
    J_v_bg = np.zeros((3, 3), dtype=np.float64)
    J_v_ba = np.zeros((3, 3), dtype=np.float64)
    J_p_bg = np.zeros((3, 3), dtype=np.float64)
    J_p_ba = np.zeros((3, 3), dtype=np.float64)
    noise = np.diag(
        [gyro_noise_density**2] * 3
        + [accel_noise_density**2] * 3
        + [gyro_random_walk**2] * 3
        + [accel_random_walk**2] * 3
    )

    for index, dt in enumerate(np.diff(times)):
        omega = 0.5 * (gyro[index] + gyro[index + 1]) - gyro_bias
        acceleration = 0.5 * (accel[index] + accel[index + 1]) - accel_bias
        acceleration_start = delta_R @ acceleration

        F = np.eye(15, dtype=np.float64)
        F[:3, 9:12] = -np.eye(3) * dt
        F[3:6, :3] = -delta_R @ skew(acceleration) * dt
        F[3:6, 12:15] = -delta_R * dt
        F[6:9, :3] = -0.5 * delta_R @ skew(acceleration) * dt**2
        F[6:9, 3:6] = np.eye(3) * dt
        F[6:9, 12:15] = -0.5 * delta_R * dt**2
        G = np.zeros((15, 12), dtype=np.float64)
        G[:3, :3] = -np.eye(3)
        G[3:6, 3:6] = -delta_R
        G[6:9, 3:6] = -0.5 * delta_R * dt
        G[9:12, 6:9] = np.eye(3)
        G[12:15, 9:12] = np.eye(3)
        covariance = F @ covariance @ F.T + G @ noise @ G.T * dt

        J_p_bg += J_v_bg * dt - 0.5 * delta_R @ skew(
            acceleration
        ) @ J_R_bg * dt**2
        J_p_ba += J_v_ba * dt - 0.5 * delta_R * dt**2
        J_v_bg += -delta_R @ skew(acceleration) @ J_R_bg * dt
        J_v_ba += -delta_R * dt
        J_R_bg -= np.eye(3) * dt
        delta_p += delta_v * dt + 0.5 * acceleration_start * dt**2
        delta_v += acceleration_start * dt
        delta_R = delta_R @ rotvec_to_rotmat(omega * dt)

    duration = float(end_time - start_time)
    return ImuPreintegration(
        dt=duration,
        delta_R=delta_R,
        delta_v=delta_v,
        delta_p=delta_p,
        covariance=covariance,
        J_R_bg=J_R_bg,
        J_v_bg=J_v_bg,
        J_v_ba=J_v_ba,
        J_p_bg=J_p_bg,
        J_p_ba=J_p_ba,
        gyro_bias=gyro_bias.copy(),
        accel_bias=accel_bias.copy(),
        sample_count=len(times),
    )
