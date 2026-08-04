from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from event_slam.core.geometry import (
    Pose,
    as_float_array,
    empty_points,
    invert_transform,
    make_transform,
)
from event_slam.core.imu import rotation_angle_deg, rotation_error_deg
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

    pnp_rotation_step_deg: float = np.nan
    imu_rotation_step_deg: float = np.nan
    pnp_imu_rotation_error_deg: float = np.nan
    imu_rotation_consistent: bool = False
    imu_rejected: bool = False

    redetected: bool = False
    reinitialized: bool = False
    depth_count: int = 0
    message: str = ""

    tracked_points_curr: np.ndarray = field(
        default_factory=lambda: empty_points(2)
    )
    pnp_points_curr: np.ndarray = field(
        default_factory=lambda: empty_points(2)
    )
    pnp_inlier_points_curr: np.ndarray = field(
        default_factory=lambda: empty_points(2)
    )


@dataclass
class StereoPnPVOSummary:
    processed_frames: int
    successful_steps: int
    failed_frames: int
    median_inliers: float
    final_position: np.ndarray


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
        R_rect_left_from_left=None,
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
        R_output_from_pnp_camera=None,
        imu_rotation_prior_max_error_deg: float = 3.0,
        imu_rotation_prior_reject_bad_pnp: bool = False,
    ) -> None:
        self.K = as_float_array(K, (3, 3), "K")
        self.P1 = as_float_array(P1, (3, 4), "P1")
        self.P2 = as_float_array(P2, (3, 4), "P2")

        if R_rect_left_from_left is None:
            self.R_rect_left_from_left = np.eye(3, dtype=np.float64)
        else:
            self.R_rect_left_from_left = as_float_array(
                R_rect_left_from_left, (3, 3), "R_rect_left_from_left"
            )

        self.R_left_from_rect_left = self.R_rect_left_from_left.T

        if R_output_from_pnp_camera is None:
            self.R_output_from_pnp_camera = np.eye(3, dtype=np.float64)
        else:
            self.R_output_from_pnp_camera = as_float_array(
                R_output_from_pnp_camera, (3, 3), "R_output_from_pnp_camera"
            )

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

        self.imu_rotation_prior_max_error_deg = float(
            imu_rotation_prior_max_error_deg
        )
        self.imu_rotation_prior_reject_bad_pnp = bool(
            imu_rotation_prior_reject_bad_pnp
        )

        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        self.reset()

    def reset(self) -> None:
        """
        Reset VO state.
        """
        self.tracker.reset()

        self.initialized = False
        self.T_W_Cleft = np.eye(4, dtype=np.float64)

        self.prev_points_3d_rect = empty_points(3, dtype=np.float64)

        self.trajectory = Trajectory()
        self.results = []

    def get_summary(self) -> StereoPnPVOSummary:
        inliers = [
            result.pnp_inlier_count
            for result in self.results
            if result.pnp_inlier_count > 0
        ]
        median_inliers = float(np.median(inliers)) if inliers else np.nan
        final_position = (
            self.results[-1].T_W_Cleft[:3, 3].copy()
            if self.results
            else np.zeros(3, dtype=np.float64)
        )

        return StereoPnPVOSummary(
            processed_frames=len(self.results),
            successful_steps=sum(result.success for result in self.results),
            failed_frames=sum(not result.success for result in self.results),
            median_inliers=median_inliers,
            final_position=final_position,
        )

    def process(
        self,
        left_rectified: np.ndarray,
        right_rectified: np.ndarray,
        timestamp: float,
        imu_rotation_prior: np.ndarray = None,
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

            self.prev_points_3d_rect = points_3d
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
            imu_rotation_prior=imu_rotation_prior,
        )

        self._append_result(result)
        return result

    def _estimate_current_pose(
        self,
        left_rectified: np.ndarray,
        right_rectified: np.ndarray,
        timestamp: float,
        tracking_result,
        imu_rotation_prior: np.ndarray = None,
    ) -> StereoPnPVOResult:
        pnp_object_points, pnp_image_points = self._make_pnp_correspondences(
            tracking_result=tracking_result,
        )

        success = False
        message = "not enough PnP correspondences"

        pnp_inlier_count = 0
        reprojection_error_mean = np.nan
        reprojection_error_median = np.nan

        pnp_rotation_step_deg = np.nan
        imu_rotation_step_deg = np.nan
        pnp_imu_rotation_error_deg = np.nan
        imu_rotation_consistent = False
        imu_rejected = False

        pnp_inlier_points_curr = empty_points(2)

        if len(pnp_object_points) >= self.min_pnp_points:
            (
                success,
                T_Ck_Cprev,
                inlier_mask,
                reprojection_error_mean,
                reprojection_error_median,
                pnp_rotation_step_deg,
                imu_rotation_step_deg,
                pnp_imu_rotation_error_deg,
                imu_rotation_consistent,
                imu_rejected,
                message,
            ) = self._solve_pnp(
                object_points=pnp_object_points,
                image_points=pnp_image_points,
                imu_rotation_prior=imu_rotation_prior,
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

        self.prev_points_3d_rect = points_3d

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
            pnp_rotation_step_deg=pnp_rotation_step_deg,
            imu_rotation_step_deg=imu_rotation_step_deg,
            pnp_imu_rotation_error_deg=pnp_imu_rotation_error_deg,
            imu_rotation_consistent=imu_rotation_consistent,
            imu_rejected=imu_rejected,
            redetected=tracking_result.redetected,
            reinitialized=False,
            depth_count=depth_count,
            message=message,
            tracked_points_curr=tracking_result.curr_points,
            pnp_points_curr=pnp_image_points.astype(np.float32),
            pnp_inlier_points_curr=pnp_inlier_points_curr,
        )

    def _make_pnp_correspondences(self, tracking_result) -> tuple:
        if len(self.prev_points_3d_rect) == 0:
            return empty_points(3, dtype=np.float64), empty_points(2)

        status_mask = tracking_result.status_mask

        if len(status_mask) != len(self.prev_points_3d_rect):
            return empty_points(3, dtype=np.float64), empty_points(2)

        object_points = self.prev_points_3d_rect[status_mask]
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
        imu_rotation_prior: np.ndarray = None,
    ) -> tuple:
        empty_inliers = np.zeros(len(object_points), dtype=np.bool_)

        if len(object_points) < 6:
            return _make_pnp_solution(
                False,
                np.eye(4, dtype=np.float64),
                empty_inliers,
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
            return _make_pnp_solution(
                False,
                np.eye(4, dtype=np.float64),
                empty_inliers,
                "solvePnPRansac cv2 error",
            )

        if not ok or inliers is None:
            return _make_pnp_solution(
                False,
                np.eye(4, dtype=np.float64),
                empty_inliers,
                "solvePnPRansac failed",
            )

        inlier_indices = inliers.reshape(-1)

        inlier_mask = np.zeros(len(object_points), dtype=np.bool_)
        inlier_mask[inlier_indices] = True

        inlier_count = int(np.count_nonzero(inlier_mask))
        inlier_ratio = inlier_count / max(1, len(object_points))

        if inlier_count < self.min_pnp_inliers:
            return _make_pnp_solution(
                False,
                np.eye(4, dtype=np.float64),
                inlier_mask,
                f"too few PnP inliers: {inlier_count}",
            )

        if inlier_ratio < self.min_pnp_inlier_ratio:
            return _make_pnp_solution(
                False,
                np.eye(4, dtype=np.float64),
                inlier_mask,
                f"PnP inlier ratio too low: {inlier_ratio:.3f}",
            )

        R_Ck_rect_Cprev_rect, _ = cv2.Rodrigues(rvec)
        t_Ck_rect_Cprev_rect = tvec.reshape(3)

        T_Ck_rect_Cprev_rect = make_transform(
            R_Ck_rect_Cprev_rect,
            t_Ck_rect_Cprev_rect,
        )

        T_Ck_Cprev = self._rectified_motion_to_left_motion(
            T_Ck_rect_Cprev_rect,
        )
        T_Ck_Cprev = self._apply_output_frame_correction(T_Ck_Cprev)

        error_mean, error_median = self._compute_reprojection_error(
            object_points=object_points[inlier_mask],
            image_points=image_points[inlier_mask],
            rvec=rvec,
            tvec=tvec,
        )

        pnp_rotation_step_deg = rotation_angle_deg(T_Ck_Cprev[:3, :3])
        imu_rotation_step_deg = np.nan
        pnp_imu_rotation_error_deg = np.nan
        imu_rotation_consistent = False

        if imu_rotation_prior is not None:
            imu_rotation_prior = np.asarray(imu_rotation_prior, dtype=np.float64)

            if imu_rotation_prior.shape == (3, 3):
                imu_rotation_step_deg = rotation_angle_deg(imu_rotation_prior)
                pnp_imu_rotation_error_deg = rotation_error_deg(
                    T_Ck_Cprev[:3, :3],
                    imu_rotation_prior,
                )
                imu_rotation_consistent = (
                    pnp_imu_rotation_error_deg
                    <= self.imu_rotation_prior_max_error_deg
                )

        if error_median > self.max_pnp_reprojection_median:
            return _make_pnp_solution(
                False,
                T_Ck_Cprev,
                inlier_mask,
                f"PnP reprojection error too high: {error_median:.3f}",
                error_mean,
                error_median,
                pnp_rotation_step_deg,
                imu_rotation_step_deg,
                pnp_imu_rotation_error_deg,
                imu_rotation_consistent,
                imu_rejected=False,
            )

        translation_step = float(np.linalg.norm(T_Ck_Cprev[:3, 3]))

        if translation_step > self.max_translation_step:
            return _make_pnp_solution(
                False,
                T_Ck_Cprev,
                inlier_mask,
                f"translation step too large: {translation_step:.3f}",
                error_mean,
                error_median,
                pnp_rotation_step_deg,
                imu_rotation_step_deg,
                pnp_imu_rotation_error_deg,
                imu_rotation_consistent,
                imu_rejected=False,
            )

        if pnp_rotation_step_deg > self.max_rotation_step_deg:
            return _make_pnp_solution(
                False,
                T_Ck_Cprev,
                inlier_mask,
                f"rotation step too large: {pnp_rotation_step_deg:.3f}",
                error_mean,
                error_median,
                pnp_rotation_step_deg,
                imu_rotation_step_deg,
                pnp_imu_rotation_error_deg,
                imu_rotation_consistent,
                imu_rejected=False,
            )

        if (
            self.imu_rotation_prior_reject_bad_pnp
            and imu_rotation_prior is not None
            and np.isfinite(pnp_imu_rotation_error_deg)
            and not imu_rotation_consistent
        ):
            return _make_pnp_solution(
                False,
                T_Ck_Cprev,
                inlier_mask,
                (
                    "PnP rejected by IMU rotation prior: "
                    f"{pnp_imu_rotation_error_deg:.3f} deg"
                ),
                error_mean,
                error_median,
                pnp_rotation_step_deg,
                imu_rotation_step_deg,
                pnp_imu_rotation_error_deg,
                imu_rotation_consistent,
                imu_rejected=True,
            )

        return _make_pnp_solution(
            True,
            T_Ck_Cprev,
            inlier_mask,
            "ok",
            error_mean,
            error_median,
            pnp_rotation_step_deg,
            imu_rotation_step_deg,
            pnp_imu_rotation_error_deg,
            imu_rotation_consistent,
            imu_rejected=False,
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

    def _rectified_motion_to_left_motion(
        self,
        T_Ck_rect_Cprev_rect: np.ndarray,
    ) -> np.ndarray:
        """
        Convert relative camera motion from rectified-left frame to original-left frame.

        R_rect_left_from_left convention:
            X_left_rect = R_rect_left_from_left @ X_left

        solvePnP estimates:
            X_Ck_rect = R_rect @ X_Cprev_rect + t_rect

        After conversion:
            X_Ck_left = R_left @ X_Cprev_left + t_left
        """
        R_rect = T_Ck_rect_Cprev_rect[:3, :3]
        t_rect = T_Ck_rect_Cprev_rect[:3, 3]

        R1 = self.R_rect_left_from_left
        R1_inv = self.R_left_from_rect_left

        R_left = R1_inv @ R_rect @ R1
        t_left = R1_inv @ t_rect

        return make_transform(R_left, t_left)

    def _apply_output_frame_correction(self, T_camera: np.ndarray) -> np.ndarray:
        C = self.R_output_from_pnp_camera

        R = C @ T_camera[:3, :3] @ C.T
        t = C @ T_camera[:3, 3]

        return make_transform(R, t)


def _make_pnp_solution(
    success: bool,
    T_Ck_Cprev: np.ndarray,
    inlier_mask: np.ndarray,
    message: str,
    reprojection_error_mean: float = np.nan,
    reprojection_error_median: float = np.nan,
    pnp_rotation_step_deg: float = np.nan,
    imu_rotation_step_deg: float = np.nan,
    pnp_imu_rotation_error_deg: float = np.nan,
    imu_rotation_consistent: bool = False,
    imu_rejected: bool = False,
) -> tuple:
    return (
        success,
        T_Ck_Cprev,
        inlier_mask,
        reprojection_error_mean,
        reprojection_error_median,
        pnp_rotation_step_deg,
        imu_rotation_step_deg,
        pnp_imu_rotation_error_deg,
        imu_rotation_consistent,
        imu_rejected,
        message,
    )
