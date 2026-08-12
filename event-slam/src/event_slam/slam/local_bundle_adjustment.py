from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from event_slam.core.geometry import (
    invert_transform,
    rotmat_to_rotvec,
    rotvec_to_rotmat,
    se3_exp,
    transform_points,
)
from event_slam.slam.imu_preintegration import preintegrate_imu


@dataclass
class LocalBAResult:
    success: bool = False
    cost_before: float = np.nan
    cost_after: float = np.nan
    evaluations: int = 0
    landmark_count: int = 0
    imu_factor_count: int = 0
    visual_cost_before: float = np.nan
    visual_cost_after: float = np.nan
    imu_cost_before: float = np.nan
    imu_cost_after: float = np.nan


@dataclass
class MotionPrior:
    keyframe_id: int
    mean: np.ndarray
    sqrt_information: np.ndarray


class LocalStereoInertialBA:
    """Fixed-lag stereo-inertial bundle adjustment over recent keyframes."""

    def __init__(
        self,
        P1,
        P2,
        R_Crect_C,
        T_C_I,
        imu_data,
        imu_calibration,
        params=None,
    ) -> None:
        params = params or {}
        self.P1 = np.asarray(P1, dtype=np.float64)
        self.P2 = np.asarray(P2, dtype=np.float64)
        self.focal_baseline = abs(float(self.P2[0, 3] - self.P1[0, 3]))
        self.R_Crect_C = np.asarray(R_Crect_C, dtype=np.float64)
        self.T_C_I = np.asarray(T_C_I, dtype=np.float64)
        self.imu_data = imu_data
        self.noise = imu_calibration
        self.window_size = int(params.get("window_size", 6))
        self.optimization_interval = max(
            1, int(params.get("optimization_interval_keyframes", 1))
        )
        self.max_landmarks = int(params.get("max_landmarks", 120))
        self.max_evaluations = int(params.get("max_evaluations", 8))
        self.pixel_sigma = float(params.get("pixel_sigma", 1.5))
        self.huber_scale = float(params.get("huber_scale", 2.0))
        self.max_pose_correction = float(params.get("max_pose_correction", 0.15))
        self.max_rotation_correction = np.radians(
            float(params.get("max_rotation_correction_deg", 5.0))
        )
        self.gravity_magnitude = float(params.get("gravity_magnitude", 9.81))
        self.stationary_gyro_threshold = float(
            params.get("stationary_gyro_threshold", 0.06)
        )
        self.stationary_accel_threshold = float(
            params.get("stationary_accel_threshold", 0.3)
        )
        self.imu_model_sigma = np.repeat(
            [
                float(params.get("imu_rotation_sigma", 0.01)),
                float(params.get("imu_velocity_sigma", 0.1)),
                float(params.get("imu_position_sigma", 0.02)),
                float(params.get("gyro_bias_sigma", 0.002)),
                float(params.get("accel_bias_sigma", 0.02)),
            ],
            3,
        )
        self.states = {}
        self.gravity_O = None
        self.initial_gyro_bias = None
        self.initial_accel_bias = np.zeros(3, dtype=np.float64)
        self.motion_prior = None

    def optimize(self, sparse_map) -> LocalBAResult:
        keyframes = sparse_map.keyframes[-self.window_size :]
        if len(keyframes) < 2:
            self._initialize(keyframes[0])
            return LocalBAResult()

        self._initialize(keyframes[0])
        observations, landmark_ids = self._select_landmarks(keyframes, sparse_map)
        if not landmark_ids:
            return LocalBAResult()

        layout = _StateLayout(len(keyframes), len(landmark_ids))
        initial_poses = [keyframe.T_O_C.copy() for keyframe in keyframes]
        x0 = self._initial_vector(layout, keyframes, landmark_ids, sparse_map)
        preintegrations = self._preintegrations(keyframes, x0, layout)
        stationary = [self._is_stationary(keyframe.timestamp) for keyframe in keyframes]
        args = (
            layout,
            keyframes,
            initial_poses,
            landmark_ids,
            observations,
            sparse_map,
            preintegrations,
            stationary,
        )
        residual_before = self._residuals(x0, *args)
        result = least_squares(
            self._residuals,
            x0,
            args=args,
            jac_sparsity=self._sparsity(
                layout, observations, landmark_ids, stationary
            ),
            loss="huber",
            f_scale=self.huber_scale,
            x_scale="jac",
            max_nfev=self.max_evaluations,
        )
        residual_after = self._residuals(result.x, *args)
        imu_rows = 15 * len(preintegrations)
        visual_rows = 4 * sum(
            len(observations[landmark_id]) for landmark_id in landmark_ids
        )
        imu_before = self._cost(residual_before[:imu_rows])
        imu_after = self._cost(residual_after[:imu_rows])
        visual_before = self._cost(
            residual_before[imu_rows : imu_rows + visual_rows]
        )
        visual_after = self._cost(
            residual_after[imu_rows : imu_rows + visual_rows]
        )
        accepted = self._accept(
            result.x,
            layout,
            residual_before,
            residual_after,
            imu_before,
            imu_after,
            visual_before,
            visual_after,
        )
        if accepted:
            self._apply(
                result.x,
                layout,
                keyframes,
                initial_poses,
                landmark_ids,
                sparse_map,
            )
            self._marginalize_oldest_motion(
                result.x,
                layout,
                keyframes,
                initial_poses,
                preintegrations[0],
                stationary[0],
            )
        return LocalBAResult(
            success=accepted,
            cost_before=self._cost(residual_before),
            cost_after=self._cost(residual_after),
            evaluations=int(result.nfev),
            landmark_count=len(landmark_ids),
            imu_factor_count=len(preintegrations),
            imu_cost_before=imu_before,
            imu_cost_after=imu_after,
            visual_cost_before=visual_before,
            visual_cost_after=visual_after,
        )

    @staticmethod
    def _cost(residual):
        return 0.5 * float(residual @ residual)

    def _initialize(self, keyframe) -> None:
        if self.gravity_O is not None:
            return
        start = self.imu_data.timestamps[0]
        end = min(self.imu_data.timestamps[-1], start + 0.75)
        mask = (self.imu_data.timestamps >= start) & (
            self.imu_data.timestamps <= end
        )
        gyro = np.median(self.imu_data.angular_velocities[mask], axis=0)
        accel = np.median(self.imu_data.linear_accelerations[mask], axis=0)
        R_O_I = (keyframe.T_O_C @ self.T_C_I)[:3, :3]
        gravity = -R_O_I @ accel
        self.gravity_O = gravity * (
            self.gravity_magnitude / max(np.linalg.norm(gravity), 1e-9)
        )
        self.initial_gyro_bias = gyro
        self.initial_accel_bias = accel + R_O_I.T @ self.gravity_O

    def _is_stationary(self, timestamp: float) -> bool:
        radius = 0.05
        mask = np.abs(self.imu_data.timestamps - timestamp) <= radius
        if np.count_nonzero(mask) < 5:
            return False
        gyro = self.imu_data.angular_velocities[mask] - self.initial_gyro_bias
        accel = self.imu_data.linear_accelerations[mask] - self.initial_accel_bias
        return bool(
            np.median(np.linalg.norm(gyro, axis=1))
            < self.stationary_gyro_threshold
            and abs(np.median(np.linalg.norm(accel, axis=1)) - self.gravity_magnitude)
            < self.stationary_accel_threshold
        )

    def _select_landmarks(self, keyframes, sparse_map) -> tuple:
        keyframe_ids = {keyframe.id for keyframe in keyframes}
        observations = {}
        for local_id, keyframe in enumerate(keyframes):
            for point_id, landmark_id in enumerate(keyframe.landmark_ids):
                if int(landmark_id) in sparse_map.landmarks:
                    observations.setdefault(int(landmark_id), []).append(
                        (local_id, point_id)
                    )
        candidates = [
            landmark_id
            for landmark_id, items in observations.items()
            if len(items) >= 2
            and sparse_map.landmarks[landmark_id].anchor_keyframe_id
            in keyframe_ids
        ]
        candidates.sort(
            key=lambda landmark_id: (
                -len(observations[landmark_id]),
                sparse_map.landmarks[landmark_id].depth_uncertainty,
            )
        )

        selected = []
        cell_counts = {}
        for landmark_id in candidates:
            local_id, point_id = observations[landmark_id][0]
            point = keyframes[local_id].points_2d[point_id]
            cell = (int(point[0] // 80), int(point[1] // 60))
            if cell_counts.get(cell, 0) >= 12:
                continue
            selected.append(landmark_id)
            cell_counts[cell] = cell_counts.get(cell, 0) + 1
            if len(selected) >= self.max_landmarks:
                break
        return observations, selected

    def _initial_vector(self, layout, keyframes, landmark_ids, sparse_map):
        x = np.zeros(layout.size, dtype=np.float64)
        positions = np.asarray([keyframe.T_O_C[:3, 3] for keyframe in keyframes])
        timestamps = np.asarray([keyframe.timestamp for keyframe in keyframes])
        velocities = np.gradient(positions, timestamps, axis=0)
        for index, keyframe in enumerate(keyframes):
            previous = self.states.get(keyframe.id)
            if previous is not None:
                velocities[index], gyro_bias, accel_bias = previous
            else:
                gyro_bias = self.initial_gyro_bias
                accel_bias = self.initial_accel_bias
            x[layout.velocity(index)] = velocities[index]
            x[layout.gyro_bias(index)] = gyro_bias
            x[layout.accel_bias(index)] = accel_bias
        x[layout.rho_start :] = [
            sparse_map.landmarks[landmark_id].inverse_depth
            for landmark_id in landmark_ids
        ]
        return x

    def _preintegrations(self, keyframes, x, layout):
        measurements = []
        for index in range(len(keyframes) - 1):
            measurements.append(
                preintegrate_imu(
                    self.imu_data,
                    keyframes[index].timestamp,
                    keyframes[index + 1].timestamp,
                    x[layout.gyro_bias(index)],
                    x[layout.accel_bias(index)],
                    self.noise.gyroscope_noise_density,
                    self.noise.accelerometer_noise_density,
                    self.noise.gyroscope_random_walk,
                    self.noise.accelerometer_random_walk,
                )
            )
        return measurements

    def _residuals(
        self,
        x,
        layout,
        keyframes,
        initial_poses,
        landmark_ids,
        observations,
        sparse_map,
        preintegrations,
        stationary,
    ):
        poses = self._poses(x, layout, initial_poses)
        residuals = []
        for index, measurement in enumerate(preintegrations):
            residuals.extend(
                self._imu_residual(x, layout, poses, index, measurement)
            )
        for rho_index, landmark_id in enumerate(landmark_ids):
            landmark = sparse_map.landmarks[landmark_id]
            anchor_local = landmark.anchor_keyframe_id - keyframes[0].id
            bearing = landmark.position_C_anchor * landmark.inverse_depth
            point_anchor = bearing / max(x[layout.rho_start + rho_index], 1e-6)
            point_W = transform_points(poses[anchor_local], point_anchor)
            for keyframe_local, point_index in observations[landmark_id]:
                point_C = transform_points(
                    invert_transform(poses[keyframe_local]), point_W
                )
                point_rect = self.R_Crect_C @ point_C
                if point_rect[2] <= 1e-6:
                    residuals.extend([100.0] * 4)
                    continue
                left = self._project(self.P1, point_rect)
                right = self._project(self.P2, point_rect)
                keyframe = keyframes[keyframe_local]
                observed_depth = (
                    self.R_Crect_C @ keyframe.points_C[point_index]
                )[2]
                sigma_disparity = (
                    keyframe.depth_uncertainties[point_index]
                    * self.focal_baseline
                    / max(observed_depth**2, 1e-12)
                )
                residuals.extend(
                    (left - keyframe.points_2d[point_index]) / self.pixel_sigma
                )
                right_error = right - keyframe.points_2d_right[point_index]
                right_error[0] /= np.hypot(self.pixel_sigma, sigma_disparity)
                right_error[1] /= self.pixel_sigma
                residuals.extend(right_error)
        for index, is_stationary in enumerate(stationary):
            if is_stationary:
                residuals.extend(x[layout.velocity(index)] / 0.02)
        residuals.extend(self._motion_prior_residual(x, layout, keyframes[0].id))
        return np.asarray(residuals, dtype=np.float64)

    def _motion_prior_residual(self, x, layout, keyframe_id):
        motion = x[layout.motion(0)]
        if self.motion_prior is not None and self.motion_prior.keyframe_id == keyframe_id:
            return self.motion_prior.sqrt_information @ (
                motion - self.motion_prior.mean
            )
        previous = self.states.get(keyframe_id)
        mean = np.concatenate(
            previous
            if previous is not None
            else (np.zeros(3), self.initial_gyro_bias, self.initial_accel_bias)
        )
        return (motion - mean) / np.repeat([0.2, 0.01, 0.1], 3)

    def _imu_residual(self, x, layout, poses, index, measurement):
        T_W_Ii = poses[index] @ self.T_C_I
        T_W_Ij = poses[index + 1] @ self.T_C_I
        R_i = T_W_Ii[:3, :3]
        p_i = T_W_Ii[:3, 3]
        R_j = T_W_Ij[:3, :3]
        p_j = T_W_Ij[:3, 3]
        v_i = x[layout.velocity(index)]
        v_j = x[layout.velocity(index + 1)]
        bg_i = x[layout.gyro_bias(index)]
        bg_j = x[layout.gyro_bias(index + 1)]
        ba_i = x[layout.accel_bias(index)]
        ba_j = x[layout.accel_bias(index + 1)]
        dt = measurement.dt
        delta_bg = bg_i - measurement.gyro_bias
        delta_ba = ba_i - measurement.accel_bias
        delta_R = measurement.delta_R @ rotvec_to_rotmat(
            measurement.J_R_bg @ delta_bg
        )
        delta_v = (
            measurement.delta_v
            + measurement.J_v_bg @ delta_bg
            + measurement.J_v_ba @ delta_ba
        )
        delta_p = (
            measurement.delta_p
            + measurement.J_p_bg @ delta_bg
            + measurement.J_p_ba @ delta_ba
        )
        residual = np.concatenate(
            (
                rotmat_to_rotvec(delta_R.T @ R_i.T @ R_j),
                R_i.T @ (v_j - v_i - self.gravity_O * dt) - delta_v,
                R_i.T
                @ (p_j - p_i - v_i * dt - 0.5 * self.gravity_O * dt**2)
                - delta_p,
                bg_j - bg_i,
                ba_j - ba_i,
            )
        )
        covariance = measurement.covariance + np.diag(self.imu_model_sigma**2)
        return np.linalg.solve(np.linalg.cholesky(covariance), residual)

    def _poses(self, x, layout, initial):
        poses = [initial[0]]
        for index in range(1, len(initial)):
            poses.append(initial[index] @ se3_exp(x[layout.pose(index)]))
        return poses

    def _apply(
        self,
        x,
        layout,
        keyframes,
        initial_poses,
        landmark_ids,
        sparse_map,
    ) -> None:
        poses = self._poses(x, layout, initial_poses)
        for index, (keyframe, pose) in enumerate(zip(keyframes, poses)):
            keyframe.T_O_C = pose
            keyframe.velocity_O = x[layout.velocity(index)].copy()
            keyframe.gyro_bias = x[layout.gyro_bias(index)].copy()
            keyframe.accel_bias = x[layout.accel_bias(index)].copy()
            self.states[keyframe.id] = (
                keyframe.velocity_O.copy(),
                keyframe.gyro_bias.copy(),
                keyframe.accel_bias.copy(),
            )
        for index, landmark_id in enumerate(landmark_ids):
            landmark = sparse_map.landmarks[landmark_id]
            rho = float(x[layout.rho_start + index])
            landmark.position_C_anchor *= landmark.inverse_depth / rho
            landmark.inverse_depth = rho

    def _marginalize_oldest_motion(
        self,
        x,
        layout,
        keyframes,
        initial_poses,
        measurement,
        stationary,
    ) -> None:
        """Carry the oldest IMU factor into a fixed-size prior for the next window."""
        poses = self._poses(x, layout, initial_poses)
        indices = np.r_[layout.motion(0), layout.motion(1)]
        point = x[indices].copy()
        args = (x, indices, layout, poses, measurement, keyframes[0].id, stationary)
        base = self._marginal_residual(point, *args)
        jacobian = np.empty((len(base), len(point)), dtype=np.float64)
        step = 1e-6
        for index in range(len(point)):
            shifted = point.copy()
            shifted[index] += step
            jacobian[:, index] = (
                self._marginal_residual(shifted, *args) - base
            ) / step

        H = jacobian.T @ jacobian + 1e-9 * np.eye(len(point))
        gradient = jacobian.T @ base
        H_00, H_01 = H[:9, :9], H[:9, 9:]
        H_10, H_11 = H[9:, :9], H[9:, 9:]
        solve_H00_H01 = np.linalg.solve(H_00, H_01)
        information = H_11 - H_10 @ solve_H00_H01
        reduced_gradient = gradient[9:] - H_10 @ np.linalg.solve(
            H_00, gradient[:9]
        )
        eigenvalues, eigenvectors = np.linalg.eigh(information)
        information = (eigenvectors * np.maximum(eigenvalues, 1e-9)) @ eigenvectors.T
        mean = point[9:] - np.linalg.solve(information, reduced_gradient)
        self.motion_prior = MotionPrior(
            keyframe_id=keyframes[1].id,
            mean=mean,
            sqrt_information=np.linalg.cholesky(information).T,
        )

    def _marginal_residual(
        self,
        values,
        x,
        indices,
        layout,
        poses,
        measurement,
        keyframe_id,
        stationary,
    ):
        trial = x.copy()
        trial[indices] = values
        parts = [self._imu_residual(trial, layout, poses, 0, measurement)]
        parts.append(self._motion_prior_residual(trial, layout, keyframe_id))
        if stationary:
            parts.append(trial[layout.velocity(0)] / 0.02)
        return np.concatenate(parts)

    def _accept(
        self,
        x,
        layout,
        before,
        after,
        imu_before,
        imu_after,
        visual_before,
        visual_after,
    ) -> bool:
        pose_increments = [
            x[layout.pose(index)] for index in range(1, layout.pose_count)
        ]
        cost_before = before @ before
        cost_after = after @ after
        return bool(
            np.all(np.isfinite(x))
            and cost_after < 0.999 * cost_before
            and imu_after <= 1.01 * imu_before
            and visual_after <= 1.01 * visual_before
            and np.all(x[layout.rho_start :] > 0.0)
            and all(
                np.linalg.norm(delta[:3]) <= self.max_pose_correction
                and np.linalg.norm(delta[3:]) <= self.max_rotation_correction
                for delta in pose_increments
            )
        )

    def _sparsity(self, layout, observations, landmark_ids, stationary):
        imu_rows = 15 * (layout.pose_count - 1)
        visual_rows = 4 * sum(len(observations[item]) for item in landmark_ids)
        stationary_rows = 3 * sum(stationary)
        prior_rows = 9
        matrix = lil_matrix((imu_rows + visual_rows + prior_rows, layout.size))
        if stationary_rows:
            matrix.resize(
                (imu_rows + visual_rows + stationary_rows + prior_rows, layout.size)
            )
        row = 0
        for index in range(layout.pose_count - 1):
            for state_id in (index, index + 1):
                if state_id > 0:
                    matrix[row : row + 15, layout.pose(state_id)] = 1
                matrix[row : row + 15, layout.motion(state_id)] = 1
            row += 15
        for rho_index, landmark_id in enumerate(landmark_ids):
            items = observations[landmark_id]
            anchor_local = min(item[0] for item in items)
            for keyframe_local, _ in items:
                if anchor_local > 0:
                    matrix[row : row + 4, layout.pose(anchor_local)] = 1
                if keyframe_local > 0:
                    matrix[row : row + 4, layout.pose(keyframe_local)] = 1
                matrix[row : row + 4, layout.rho_start + rho_index] = 1
                row += 4
        for index, is_stationary in enumerate(stationary):
            if is_stationary:
                matrix[row : row + 3, layout.velocity(index)] = 1
                row += 3
        matrix[row : row + 9, layout.motion(0)] = 1
        return matrix.tocsr()

    @staticmethod
    def _project(P, point_rect):
        homogeneous = P @ np.append(point_rect, 1.0)
        return homogeneous[:2] / homogeneous[2]


class _StateLayout:
    def __init__(self, pose_count: int, landmark_count: int) -> None:
        self.pose_count = pose_count
        self.pose_size = 6 * (pose_count - 1)
        self.velocity_start = self.pose_size
        self.gyro_start = self.velocity_start + 3 * pose_count
        self.accel_start = self.gyro_start + 3 * pose_count
        self.rho_start = self.accel_start + 3 * pose_count
        self.size = self.rho_start + landmark_count

    def pose(self, index):
        start = 6 * (index - 1)
        return slice(start, start + 6)

    def velocity(self, index):
        start = self.velocity_start + 3 * index
        return slice(start, start + 3)

    def gyro_bias(self, index):
        start = self.gyro_start + 3 * index
        return slice(start, start + 3)

    def accel_bias(self, index):
        start = self.accel_start + 3 * index
        return slice(start, start + 3)

    def motion(self, index):
        return np.r_[
            np.arange(self.velocity(index).start, self.velocity(index).stop),
            np.arange(self.gyro_bias(index).start, self.gyro_bias(index).stop),
            np.arange(self.accel_bias(index).start, self.accel_bias(index).stop),
        ]
