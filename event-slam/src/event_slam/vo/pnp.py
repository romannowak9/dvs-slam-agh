from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from event_slam.core.geometry import make_transform
from event_slam.core.imu import rotation_angle_deg, rotation_error_deg


@dataclass
class PnPSolution:
    success: bool
    T_Ck_Cprev: np.ndarray
    inlier_mask: np.ndarray
    message: str
    reprojection_error_mean: float = np.nan
    reprojection_error_median: float = np.nan
    pnp_rotation_step_deg: float = np.nan
    imu_rotation_step_deg: float = np.nan
    pnp_imu_rotation_error_deg: float = np.nan
    imu_rotation_consistent: bool = False
    imu_rejected: bool = False

    @classmethod
    def failure(cls, point_count: int, message: str) -> "PnPSolution":
        return cls(
            success=False,
            T_Ck_Cprev=np.eye(4, dtype=np.float64),
            inlier_mask=np.zeros(int(point_count), dtype=np.bool_),
            message=message,
        )


class PnPSolver:
    """Estimate and validate camera motion from 3D-2D correspondences."""

    def __init__(
        self,
        K: np.ndarray,
        min_points: int = 20,
        min_inliers: int = 30,
        min_inlier_ratio: float = 0.15,
        max_reprojection_median: float = 3.0,
        max_translation_step: float = 0.5,
        max_rotation_step_deg: float = 15.0,
        reprojection_error: float = 3.0,
        confidence: float = 0.999,
        iterations: int = 100,
        flags: int = cv2.SOLVEPNP_ITERATIVE,
        imu_max_error_deg: float = 3.0,
        reject_imu_inconsistent: bool = False,
        use_imu_rotation: bool = False,
    ) -> None:
        self.K = np.asarray(K, dtype=np.float64)
        self.min_points = max(int(min_points), 6)
        self.min_inliers = int(min_inliers)
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.max_reprojection_median = float(max_reprojection_median)
        self.max_translation_step = float(max_translation_step)
        self.max_rotation_step_deg = float(max_rotation_step_deg)
        self.reprojection_error = float(reprojection_error)
        self.confidence = float(confidence)
        self.iterations = int(iterations)
        self.flags = int(flags)
        self.imu_max_error_deg = float(imu_max_error_deg)
        self.reject_imu_inconsistent = bool(reject_imu_inconsistent)
        self.use_imu_rotation = bool(use_imu_rotation)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        self.rng = np.random.RandomState(0)

    def solve(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
        motion_converter,
        imu_rotation_prior: np.ndarray = None,
        refinement_T_Crect_object: np.ndarray = None,
        fixed_rotation_Crect_object: np.ndarray = None,
    ) -> PnPSolution:
        point_count = len(object_points)
        if point_count < 6:
            return PnPSolution.failure(
                point_count,
                f"too few PnP points: {point_count}",
            )

        imu_rotation = None
        if imu_rotation_prior is not None:
            candidate = np.asarray(imu_rotation_prior, dtype=np.float64)
            if candidate.shape == (3, 3):
                imu_rotation = candidate
        fixed_rotation = None
        if (
            self.use_imu_rotation
            and imu_rotation is not None
            and fixed_rotation_Crect_object is not None
        ):
            fixed_rotation = np.asarray(
                fixed_rotation_Crect_object,
                dtype=np.float64,
            ).reshape(3, 3)

        if fixed_rotation is not None:
            tvec, inlier_mask = self._ransac_translation(
                object_points,
                image_points,
                fixed_rotation,
            )
            if tvec is None:
                return PnPSolution.failure(
                    point_count,
                    "fixed-rotation RANSAC failed",
                )
            R_Crect_object = fixed_rotation
            rvec, _ = cv2.Rodrigues(R_Crect_object)
        else:
            try:
                ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                    objectPoints=object_points.reshape(-1, 1, 3),
                    imagePoints=image_points.reshape(-1, 1, 2),
                    cameraMatrix=self.K,
                    distCoeffs=self.dist_coeffs,
                    iterationsCount=self.iterations,
                    reprojectionError=self.reprojection_error,
                    confidence=self.confidence,
                    flags=self.flags,
                )
            except cv2.error:
                return PnPSolution.failure(
                    point_count,
                    "solvePnPRansac cv2 error",
                )

            if not ok or inliers is None:
                return PnPSolution.failure(point_count, "solvePnPRansac failed")

            inlier_mask = np.zeros(point_count, dtype=np.bool_)
            inlier_mask[inliers.reshape(-1)] = True
            R_Crect_object, _ = cv2.Rodrigues(rvec)

        inlier_count = int(np.count_nonzero(inlier_mask))

        if inlier_count < self.min_inliers:
            solution = PnPSolution.failure(
                point_count,
                f"too few PnP inliers: {inlier_count}",
            )
            solution.inlier_mask = inlier_mask
            return solution

        inlier_ratio = inlier_count / max(1, point_count)
        if inlier_ratio < self.min_inlier_ratio:
            solution = PnPSolution.failure(
                point_count,
                f"PnP inlier ratio too low: {inlier_ratio:.3f}",
            )
            solution.inlier_mask = inlier_mask
            return solution

        if fixed_rotation is None and refinement_T_Crect_object is not None:
            rvec_guess, _ = cv2.Rodrigues(refinement_T_Crect_object[:3, :3])
            try:
                refined, rvec_refined, tvec_refined = cv2.solvePnP(
                    objectPoints=object_points[inlier_mask],
                    imagePoints=image_points[inlier_mask],
                    cameraMatrix=self.K,
                    distCoeffs=self.dist_coeffs,
                    rvec=rvec_guess,
                    tvec=refinement_T_Crect_object[:3, 3].reshape(3, 1),
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
            except cv2.error:
                refined = False
            if refined:
                rvec, tvec = rvec_refined, tvec_refined

        T_Ck_Cprev = motion_converter(
            make_transform(R_Crect_object, tvec.reshape(3))
        )
        rotation_step_deg = rotation_angle_deg(T_Ck_Cprev[:3, :3])
        imu_rotation_step_deg = np.nan
        imu_error_deg = np.nan
        imu_consistent = False

        if imu_rotation is not None:
            imu_rotation_step_deg = rotation_angle_deg(imu_rotation)
            imu_error_deg = rotation_error_deg(
                T_Ck_Cprev[:3, :3],
                imu_rotation,
            )
            imu_consistent = imu_error_deg <= self.imu_max_error_deg

        error_mean, error_median = self._reprojection_error(
            object_points[inlier_mask],
            image_points[inlier_mask],
            rvec,
            tvec,
        )

        solution = PnPSolution(
            success=False,
            T_Ck_Cprev=T_Ck_Cprev,
            inlier_mask=inlier_mask,
            message="",
            reprojection_error_mean=error_mean,
            reprojection_error_median=error_median,
            pnp_rotation_step_deg=rotation_step_deg,
            imu_rotation_step_deg=imu_rotation_step_deg,
            pnp_imu_rotation_error_deg=imu_error_deg,
            imu_rotation_consistent=imu_consistent,
        )

        if error_median > self.max_reprojection_median:
            solution.message = (
                f"PnP reprojection error too high: {error_median:.3f}"
            )
        elif np.linalg.norm(T_Ck_Cprev[:3, 3]) > self.max_translation_step:
            translation = np.linalg.norm(T_Ck_Cprev[:3, 3])
            solution.message = f"translation step too large: {translation:.3f}"
        elif rotation_step_deg > self.max_rotation_step_deg:
            solution.message = (
                f"rotation step too large: {rotation_step_deg:.3f}"
            )
        elif (
            self.reject_imu_inconsistent
            and imu_rotation_prior is not None
            and np.isfinite(imu_error_deg)
            and not imu_consistent
        ):
            solution.message = (
                f"PnP rejected by IMU rotation prior: {imu_error_deg:.3f} deg"
            )
            solution.imu_rejected = True
        else:
            solution.success = True
            solution.message = "ok"

        return solution

    def _ransac_translation(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
        rotation: np.ndarray,
    ) -> tuple:
        """Select inliers while estimating only translation for a fixed rotation."""
        normalized = cv2.undistortPoints(
            image_points.reshape(-1, 1, 2),
            self.K,
            self.dist_coeffs,
        ).reshape(-1, 2)
        rotated = (rotation @ object_points.T).T
        best_mask = np.zeros(len(object_points), dtype=np.bool_)
        for _ in range(self.iterations):
            sample = self.rng.choice(len(object_points), 3, replace=False)
            translation = self._linear_translation(
                rotated[sample],
                normalized[sample],
            )
            mask = self._translation_inliers(
                rotated,
                image_points,
                translation,
            )
            if np.count_nonzero(mask) > np.count_nonzero(best_mask):
                best_mask = mask

        if np.count_nonzero(best_mask) < 3:
            return None, best_mask
        translation = self._solve_translation(
            object_points[best_mask],
            image_points[best_mask],
            rotation,
        )
        return translation, self._translation_inliers(
            rotated,
            image_points,
            translation,
        )

    def _translation_inliers(
        self,
        rotated_points: np.ndarray,
        image_points: np.ndarray,
        translation: np.ndarray,
    ) -> np.ndarray:
        points_C = rotated_points + translation.reshape(1, 3)
        projected = points_C @ self.K.T
        valid = np.isfinite(points_C).all(axis=1) & (points_C[:, 2] > 0.0)
        errors = np.full(len(points_C), np.inf, dtype=np.float64)
        image = projected[valid, :2] / projected[valid, 2:3]
        errors[valid] = np.linalg.norm(image - image_points[valid], axis=1)
        return errors <= self.reprojection_error

    def _solve_translation(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
        rotation: np.ndarray,
    ) -> np.ndarray:
        """Estimate three translation variables for a fixed camera rotation."""
        normalized = cv2.undistortPoints(
            image_points.reshape(-1, 1, 2),
            self.K,
            self.dist_coeffs,
        ).reshape(-1, 2)
        rotated = (rotation @ object_points.T).T
        translation = self._linear_translation(rotated, normalized)

        for _ in range(5):
            points_C = rotated + translation
            inverse_depth = 1.0 / points_C[:, 2]
            residual = points_C[:, :2] * inverse_depth[:, None] - normalized
            J = np.zeros((2 * len(points_C), 3), dtype=np.float64)
            J[0::2, 0] = inverse_depth
            J[1::2, 1] = inverse_depth
            J[0::2, 2] = -points_C[:, 0] * inverse_depth**2
            J[1::2, 2] = -points_C[:, 1] * inverse_depth**2
            step, _, _, _ = np.linalg.lstsq(
                J,
                -residual.reshape(-1),
                rcond=None,
            )
            translation += step
            if np.linalg.norm(step) < 1e-9:
                break
        return translation.reshape(3, 1)

    @staticmethod
    def _linear_translation(
        rotated_points: np.ndarray,
        normalized_points: np.ndarray,
    ) -> np.ndarray:
        """Linear translation estimate used by RANSAC and final refinement."""
        A = np.zeros((2 * len(rotated_points), 3), dtype=np.float64)
        A[0::2, 0] = 1.0
        A[1::2, 1] = 1.0
        A[0::2, 2] = -normalized_points[:, 0]
        A[1::2, 2] = -normalized_points[:, 1]
        b = np.empty(2 * len(rotated_points), dtype=np.float64)
        b[0::2] = (
            normalized_points[:, 0] * rotated_points[:, 2]
            - rotated_points[:, 0]
        )
        b[1::2] = (
            normalized_points[:, 1] * rotated_points[:, 2]
            - rotated_points[:, 1]
        )
        translation, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return translation

    def _reprojection_error(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> tuple:
        projected, _ = cv2.projectPoints(
            object_points.reshape(-1, 1, 3),
            rvec,
            tvec,
            self.K,
            self.dist_coeffs,
        )
        error = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
        return float(np.mean(error)), float(np.median(error))
