from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from event_slam.core.geometry import (
    Pose,
    invert_transform,
    make_transform,
    rotmat_to_quat_xyzw,
)
from event_slam.core.trajectory import Trajectory
from event_slam.vo.feature_tracker import FeatureTracker
from event_slam.vo.stereo_depth import StereoDepthEstimator


@dataclass
class StereoPnPVOResult:
    """
    Result of processing one rectified stereo frame pair.
    """

    timestamp: float
    success: bool
    initialized: bool
    T_W_Cleft: np.ndarray

    track_count: int = 0
    pnp_point_count: int = 0
    pnp_inlier_count: int = 0

    reprojection_error_mean: float = np.nan
    reprojection_error_median: float = np.nan

    redetected: bool = False
    reinitialized: bool = False
    depth_count: int = 0
    message: str = ""

    tracked_points_curr: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float32)
    )
    pnp_points_curr: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float32)
    )
    pnp_inlier_points_curr: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float32)
    )


class StereoPnPVO:
    """
    ELOPE-like stereo visual odometry frontend.

    The class estimates camera motion from sparse stereo depth and temporal
    feature tracking:

        1. detect and track features in the left rectified image,
        2. triangulate sparse 3D points from stereo,
        3. use previous 3D points and current 2D observations in solvePnPRansac,
        4. integrate the camera pose T_W_Cleft.

    Convention:
        T_A_B maps points from frame B to frame A.

    OpenCV solvePnP returns:
        X_Ck = R * X_Cprev + t

    Therefore the estimated relative transform is:
        T_Ck_Cprev

    Since poses are stored as T_W_C, pose integration is:
        T_W_Ck = T_W_Cprev @ inverse(T_Ck_Cprev)
    """

    def __init__(
        self,
        K: np.ndarray,
        P1: np.ndarray,
        P2: np.ndarray,
        feature_tracker_params=None,
        stereo_depth_params=None,
        min_pnp_points: int = 20,
        min_pnp_inliers: int = 30,
        min_pnp_inlier_ratio: float = 0.15,
        max_pnp_reprojection_median: float = 3.0,
        max_translation_step: float = 0.5,
        max_rotation_step_deg: float = 15.0,
        pnp_reprojection_error: float = 3.0,
        pnp_confidence: float = 0.999,
        pnp_iterations: int = 100,
        pnp_flags: int = cv2.SOLVEPNP_ITERATIVE,
    ) -> None:
        self.K = np.asarray(K, dtype=np.float64)
        self.P1 = np.asarray(P1, dtype=np.float64)
        self.P2 = np.asarray(P2, dtype=np.float64)

        if self.K.shape != (3, 3):
            raise ValueError(f"K must have shape (3, 3), got {self.K.shape}")

        if self.P1.shape != (3, 4):
            raise ValueError(f"P1 must have shape (3, 4), got {self.P1.shape}")

        if self.P2.shape != (3, 4):
            raise ValueError(f"P2 must have shape (3, 4), got {self.P2.shape}")

        feature_tracker_params = feature_tracker_params or {}
        stereo_depth_params = stereo_depth_params or {}

        self.tracker = FeatureTracker(**feature_tracker_params)

        self.depth_estimator = StereoDepthEstimator(
            P1=self.P1,
            P2=self.P2,
            **stereo_depth_params,
        )

        self.min_pnp_points = max(int(min_pnp_points), 6)
        self.min_pnp_inliers = int(min_pnp_inliers)
        self.min_pnp_inlier_ratio = float(min_pnp_inlier_ratio)

        self.max_pnp_reprojection_median = float(max_pnp_reprojection_median)
        self.max_translation_step = float(max_translation_step)
        self.max_rotation_step_deg = float(max_rotation_step_deg)

        self.pnp_reprojection_error = float(pnp_reprojection_error)
        self.pnp_confidence = float(pnp_confidence)
        self.pnp_iterations = int(pnp_iterations)
        self.pnp_flags = int(pnp_flags)

        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        self.reset()

    def reset(self) -> None:
        """
        Reset VO state.
        """
        self.tracker.reset()

        self.initialized = False
        self.T_W_Cleft = np.eye(4, dtype=np.float64)

        self.prev_points_3d = np.empty((0, 3), dtype=np.float64)

        self.trajectory = Trajectory()
        self.results = []

    def process(
        self,
        left_rectified: np.ndarray,
        right_rectified: np.ndarray,
        timestamp: float,
    ) -> StereoPnPVOResult:
        """
        Process one rectified stereo pair.
        """
        tracking_result = self.tracker.process(left_rectified)

        if not self.initialized:
            points_3d, depth_count = self._compute_depth_for_active_points(
                left_rectified=left_rectified,
                right_rectified=right_rectified,
                active_points=tracking_result.active_points,
            )

            self.prev_points_3d = points_3d
            self.initialized = True

            result = StereoPnPVOResult(
                timestamp=float(timestamp),
                success=True,
                initialized=True,
                T_W_Cleft=self.T_W_Cleft.copy(),
                redetected=tracking_result.redetected,
                reinitialized=True,
                depth_count=depth_count,
                message="initialized",
            )

            self._append_result(result)
            return result

        result = self._estimate_current_pose(
            left_rectified=left_rectified,
            right_rectified=right_rectified,
            timestamp=float(timestamp),
            tracking_result=tracking_result,
        )

        self._append_result(result)
        return result

    def save_csv(self, path) -> None:
        """
        Save trajectory together with basic VO diagnostics to CSV.

        This is intentionally separate from Trajectory, because it stores VO-specific
        status fields, not only poses.
        """
        with open(path, "w", encoding="utf-8") as file:
            file.write(
                "timestamp,success,tx,ty,tz,qx,qy,qz,qw,"
                "track_count,pnp_point_count,pnp_inlier_count,"
                "reprojection_error_mean,reprojection_error_median,message\n"
            )

            for result in self.results:
                T_W_C = result.T_W_Cleft
                t = T_W_C[:3, 3]
                qx, qy, qz, qw = rotmat_to_quat_xyzw(T_W_C[:3, :3])

                file.write(
                    f"{result.timestamp:.9f},"
                    f"{int(result.success)},"
                    f"{t[0]:.9f},{t[1]:.9f},{t[2]:.9f},"
                    f"{qx:.9f},{qy:.9f},{qz:.9f},{qw:.9f},"
                    f"{result.track_count},"
                    f"{result.pnp_point_count},"
                    f"{result.pnp_inlier_count},"
                    f"{result.reprojection_error_mean:.9f},"
                    f"{result.reprojection_error_median:.9f},"
                    f"{_csv_safe(result.message)}\n"
                )

    def _estimate_current_pose(
        self,
        left_rectified: np.ndarray,
        right_rectified: np.ndarray,
        timestamp: float,
        tracking_result,
    ) -> StereoPnPVOResult:
        pnp_object_points, pnp_image_points = self._make_pnp_correspondences(
            tracking_result=tracking_result,
        )

        success = False
        message = "not enough PnP correspondences"

        pnp_inlier_count = 0
        reprojection_error_mean = np.nan
        reprojection_error_median = np.nan

        pnp_inlier_points_curr = np.empty((0, 2), dtype=np.float32)

        if len(pnp_object_points) >= self.min_pnp_points:
            (
                success,
                T_Ck_Cprev,
                inlier_mask,
                reprojection_error_mean,
                reprojection_error_median,
                message,
            ) = self._solve_pnp(
                object_points=pnp_object_points,
                image_points=pnp_image_points,
            )

            pnp_inlier_count = int(np.count_nonzero(inlier_mask))
            pnp_inlier_points_curr = pnp_image_points[inlier_mask].astype(np.float32)

            if success:
                self.T_W_Cleft = self.T_W_Cleft @ invert_transform(T_Ck_Cprev)

        points_3d, depth_count = self._compute_depth_for_active_points(
            left_rectified=left_rectified,
            right_rectified=right_rectified,
            active_points=tracking_result.active_points,
        )

        self.prev_points_3d = points_3d

        return StereoPnPVOResult(
            timestamp=timestamp,
            success=success,
            initialized=True,
            T_W_Cleft=self.T_W_Cleft.copy(),
            track_count=tracking_result.track_count,
            pnp_point_count=len(pnp_object_points),
            pnp_inlier_count=pnp_inlier_count,
            reprojection_error_mean=reprojection_error_mean,
            reprojection_error_median=reprojection_error_median,
            redetected=tracking_result.redetected,
            reinitialized=False,
            depth_count=depth_count,
            message=message,
            tracked_points_curr=tracking_result.curr_points,
            pnp_points_curr=pnp_image_points.astype(np.float32),
            pnp_inlier_points_curr=pnp_inlier_points_curr,
        )

    def _make_pnp_correspondences(self, tracking_result) -> tuple:
        if len(self.prev_points_3d) == 0:
            return _empty_points3(), _empty_points2()

        status_mask = tracking_result.status_mask

        if len(status_mask) != len(self.prev_points_3d):
            return _empty_points3(), _empty_points2()

        object_points = self.prev_points_3d[status_mask]
        image_points = tracking_result.curr_points

        finite_mask = np.isfinite(object_points).all(axis=1)

        positive_depth_mask = np.zeros(len(object_points), dtype=np.bool_)
        positive_depth_mask[finite_mask] = object_points[finite_mask, 2] > 0.0

        valid_mask = finite_mask & positive_depth_mask

        return (
            object_points[valid_mask].astype(np.float64),
            image_points[valid_mask].astype(np.float64),
        )

    def _solve_pnp(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
    ) -> tuple:
        if len(object_points) < 6:
            return (
                False,
                np.eye(4, dtype=np.float64),
                np.zeros(len(object_points), dtype=np.bool_),
                np.nan,
                np.nan,
                f"too few PnP points: {len(object_points)}",
            )

        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                objectPoints=object_points.reshape(-1, 1, 3),
                imagePoints=image_points.reshape(-1, 1, 2),
                cameraMatrix=self.K,
                distCoeffs=self.dist_coeffs,
                iterationsCount=self.pnp_iterations,
                reprojectionError=self.pnp_reprojection_error,
                confidence=self.pnp_confidence,
                flags=self.pnp_flags,
            )
        except cv2.error:
            return (
                False,
                np.eye(4, dtype=np.float64),
                np.zeros(len(object_points), dtype=np.bool_),
                np.nan,
                np.nan,
                "solvePnPRansac cv2 error",
            )

        if not ok or inliers is None:
            return (
                False,
                np.eye(4, dtype=np.float64),
                np.zeros(len(object_points), dtype=np.bool_),
                np.nan,
                np.nan,
                "solvePnPRansac failed",
            )

        inlier_indices = inliers.reshape(-1)

        inlier_mask = np.zeros(len(object_points), dtype=np.bool_)
        inlier_mask[inlier_indices] = True

        inlier_count = int(np.count_nonzero(inlier_mask))
        inlier_ratio = inlier_count / max(1, len(object_points))

        if inlier_count < self.min_pnp_inliers:
            return (
                False,
                np.eye(4, dtype=np.float64),
                inlier_mask,
                np.nan,
                np.nan,
                f"too few PnP inliers: {inlier_count}",
            )

        if inlier_ratio < self.min_pnp_inlier_ratio:
            return (
                False,
                np.eye(4, dtype=np.float64),
                inlier_mask,
                np.nan,
                np.nan,
                f"PnP inlier ratio too low: {inlier_ratio:.3f}",
            )

        R_Ck_Cprev, _ = cv2.Rodrigues(rvec)
        t_Ck_Cprev = tvec.reshape(3)

        T_Ck_Cprev = make_transform(R_Ck_Cprev, t_Ck_Cprev)

        error_mean, error_median = self._compute_reprojection_error(
            object_points=object_points[inlier_mask],
            image_points=image_points[inlier_mask],
            rvec=rvec,
            tvec=tvec,
        )

        if error_median > self.max_pnp_reprojection_median:
            return (
                False,
                np.eye(4, dtype=np.float64),
                inlier_mask,
                error_mean,
                error_median,
                f"PnP reprojection error too high: {error_median:.3f}",
            )

        translation_step = float(np.linalg.norm(t_Ck_Cprev))
        rotation_step_deg = _rotation_angle_deg(R_Ck_Cprev)

        if translation_step > self.max_translation_step:
            return (
                False,
                np.eye(4, dtype=np.float64),
                inlier_mask,
                error_mean,
                error_median,
                f"translation step too large: {translation_step:.3f}",
            )

        if rotation_step_deg > self.max_rotation_step_deg:
            return (
                False,
                np.eye(4, dtype=np.float64),
                inlier_mask,
                error_mean,
                error_median,
                f"rotation step too large: {rotation_step_deg:.3f}",
            )

        return (
            True,
            T_Ck_Cprev,
            inlier_mask,
            error_mean,
            error_median,
            "ok",
        )

    def _compute_reprojection_error(
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

        projected = projected.reshape(-1, 2)
        error = np.linalg.norm(projected - image_points, axis=1)

        return float(np.mean(error)), float(np.median(error))

    def _compute_depth_for_active_points(
        self,
        left_rectified: np.ndarray,
        right_rectified: np.ndarray,
        active_points: np.ndarray,
    ) -> tuple:
        depth_result = self.depth_estimator.estimate(
            left_img=left_rectified,
            right_img=right_rectified,
            left_points=active_points,
        )

        points_3d = np.full((len(active_points), 3), np.nan, dtype=np.float64)

        if len(active_points) > 0 and len(depth_result.points_3d_left_camera) > 0:
            points_3d[depth_result.valid_mask] = depth_result.points_3d_left_camera

        return points_3d, depth_result.stats.triangulated_count

    def _append_result(self, result: StereoPnPVOResult) -> None:
        self.trajectory.append(
            timestamp=result.timestamp,
            pose=Pose.from_matrix(result.T_W_Cleft),
        )
        self.results.append(result)


def _empty_points2() -> np.ndarray:
    return np.empty((0, 2), dtype=np.float32)


def _empty_points3() -> np.ndarray:
    return np.empty((0, 3), dtype=np.float64)


def _rotation_angle_deg(R: np.ndarray) -> float:
    cos_angle = 0.5 * (float(np.trace(R)) - 1.0)
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))

    return float(np.degrees(np.arccos(cos_angle)))


def _csv_safe(value: str) -> str:
    return str(value).replace(",", ";").replace("\n", " ")