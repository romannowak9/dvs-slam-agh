from __future__ import annotations

import cv2
import numpy as np

from event_slam.core.geometry import (
    Pose,
    as_float_array,
    empty_points,
    invert_transform,
    make_transform,
    orthonormalize_rotation,
)
from event_slam.core.imu import rotation_angle_deg
from event_slam.core.trajectory import Trajectory
from event_slam.slam.local_matching import LocalMapMatcher, MapCorrespondences
from event_slam.slam.map import SparseMap
from event_slam.slam.place_recognition import PlaceRecognizer
from event_slam.slam.pose_graph import PoseGraph
from event_slam.slam.results import StereoPnPSLAMResult, StereoPnPSLAMSummary
from event_slam.vo.feature_tracker import FeatureTracker
from event_slam.vo.pnp import PnPSolution, PnPSolver
from event_slam.vo.stereo_depth import StereoDepthEstimator


class StereoPnPSLAM:
    """
    Stereo pose estimator with a persistent sparse map.

    The current pose is estimated first from world landmarks and current image
    points. The original frame-to-frame stereo PnP remains the fallback while
    the camera explores an unmapped area.

    Convention:
        T_A_B maps points from frame B to frame A.

    Both PnP paths are converted to the same relative transform T_Ck_Cprev.
    Since poses are stored as T_W_C, the integration is:
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
        slam_params=None,
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
        self.pnp = PnPSolver(
            K=self.K,
            min_points=min_pnp_points,
            min_inliers=min_pnp_inliers,
            min_inlier_ratio=min_pnp_inlier_ratio,
            max_reprojection_median=max_pnp_reprojection_median,
            max_translation_step=max_translation_step,
            max_rotation_step_deg=max_rotation_step_deg,
            reprojection_error=pnp_reprojection_error,
            confidence=pnp_confidence,
            iterations=pnp_iterations,
            flags=pnp_flags,
            imu_max_error_deg=imu_rotation_prior_max_error_deg,
            reject_imu_inconsistent=imu_rotation_prior_reject_bad_pnp,
        )

        slam_params = slam_params or {}
        self.slam_enabled = bool(slam_params.get("enabled", False))
        self.keyframe_min_frame_gap = int(slam_params.get("min_frame_gap", 5))
        self.keyframe_max_frame_gap = int(slam_params.get("max_frame_gap", 30))
        self.keyframe_translation_depth_ratio = float(
            slam_params.get("translation_depth_ratio", 0.15)
        )
        self.keyframe_rotation_deg = float(slam_params.get("rotation_deg", 10.0))
        self.keyframe_tracked_landmark_ratio = float(
            slam_params.get("tracked_landmark_ratio", 0.6)
        )
        self.max_landmark_depth = float(
            slam_params.get("max_landmark_depth", 10.0)
        )
        self.local_matcher = LocalMapMatcher(
            K=self.K,
            descriptor_extractor=self.tracker.describe,
            keyframe_count=slam_params.get("local_keyframes", 5),
            descriptor_ratio=slam_params.get("descriptor_ratio", 0.75),
            reprojection_gate_px=slam_params.get("reprojection_gate_px", 15.0),
        )

        self.R_C_Crect = (
            self.R_output_from_pnp_camera @ self.R_left_from_rect_left
        )
        self.T_C_Crect = make_transform(self.R_C_Crect, np.zeros(3))
        self.T_Crect_C = invert_transform(self.T_C_Crect)

        loop_cfg = slam_params.get("loop_closure", {})
        graph_cfg = slam_params.get("pose_graph", {})
        relocalization_cfg = slam_params.get("relocalization", {})
        self.loop_enabled = self.slam_enabled and bool(
            loop_cfg.get("enabled", True)
        )
        self.loop_min_keyframe_gap = int(loop_cfg.get("min_keyframe_gap", 20))
        self.loop_candidates_per_frame = int(loop_cfg.get("candidates_per_frame", 15))
        self.loop_candidate_check_index = 0
        self.loop_min_interval = max(
            1,
            int(loop_cfg.get("min_keyframes_between_loops", 10)),
        )
        self.relocalization_enabled = self.slam_enabled and bool(
            relocalization_cfg.get("enabled", True)
        )
        self.relocalization_failure_count = int(
            relocalization_cfg.get("failure_count", 3)
        )
        self.pose_graph_params = {
            "translation_weight": graph_cfg.get("translation_weight", 1.0),
            "rotation_weight": graph_cfg.get("rotation_weight", 1.0),
            "huber_scale": graph_cfg.get("huber_scale", 1.0),
            "max_evaluations": graph_cfg.get("max_evaluations", 100),
        }
        self.place_recognizer = PlaceRecognizer(
            K=self.K,
            R_Crect_C=self.T_Crect_C[:3, :3],
            motion_converter=self._rectified_candidate_to_output_motion,
            descriptor_ratio=loop_cfg.get("descriptor_ratio", 0.75),
            min_matches=loop_cfg.get("min_matches", 30),
            max_candidates=loop_cfg.get("max_candidates", 3),
            min_inliers=loop_cfg.get("min_inliers", 20),
            min_inlier_ratio=loop_cfg.get("min_inlier_ratio", 0.3),
            max_reprojection_median=loop_cfg.get(
                "max_reprojection_median", 3.0
            ),
            min_scale_error_reduction=loop_cfg.get(
                "min_scale_error_reduction", 0.2
            ),
            pnp_reprojection_error=pnp_reprojection_error,
            pnp_confidence=pnp_confidence,
            pnp_iterations=pnp_iterations,
        )

        self.reset()

    def reset(self) -> None:
        """Reset tracking, pose, trajectory and map state."""
        self.tracker.reset()

        self.initialized = False
        self.T_W_Cleft = np.eye(4, dtype=np.float64)

        self.prev_points_3d_rect = empty_points(3, dtype=np.float64)

        self.trajectory = Trajectory()
        self.results = []
        self.map = SparseMap() if self.slam_enabled else None
        self.pose_graph = PoseGraph(**self.pose_graph_params) if self.map else None
        self.consecutive_failures = 0
        self.last_graph_cost_before = np.nan
        self.last_graph_cost_after = np.nan
        self.last_loop_keyframe_id = -self.loop_min_interval

    def get_summary(self) -> StereoPnPSLAMSummary:
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

        return StereoPnPSLAMSummary(
            processed_frames=len(self.results),
            successful_steps=sum(result.success for result in self.results),
            failed_frames=sum(not result.success for result in self.results),
            median_inliers=median_inliers,
            final_position=final_position,
            keyframe_count=len(self.map.keyframes) if self.map is not None else 0,
            landmark_count=len(self.map.landmarks) if self.map is not None else 0,
            accepted_loop_count=sum(result.loop_accepted for result in self.results),
            relocalization_count=sum(result.relocalized for result in self.results),
            graph_cost_before=self.last_graph_cost_before,
            graph_cost_after=self.last_graph_cost_after,
        )

    def process(
        self,
        left_rectified: np.ndarray,
        right_rectified: np.ndarray,
        timestamp: float,
        imu_rotation_prior: np.ndarray = None,
    ) -> StereoPnPSLAMResult:
        """
        Process one rectified stereo pair.
        """
        tracking_result = self.tracker.process(left_rectified)
        if self.map is not None:
            self.map.retain_tracks(tracking_result.active_ids)

        if not self.initialized:
            points_3d, depth_count = self._compute_depth_for_active_points(
                left_rectified=left_rectified,
                right_rectified=right_rectified,
                active_points=tracking_result.active_points,
            )

            self.prev_points_3d_rect = points_3d
            self.initialized = True

            result = StereoPnPSLAMResult(
                timestamp=float(timestamp),
                success=True,
                initialized=True,
                T_W_Cleft=self.T_W_Cleft.copy(),
                pose_source="initialization",
                new_feature_count=tracking_result.detected_count,
                reinitialized=True,
                depth_count=depth_count,
                message="initialized",
            )

            self._set_frame_reference(result)
            self._update_map(left_rectified, tracking_result, result)
            self._append_result(result)
            return result

        result = self._estimate_current_pose(
            left_rectified=left_rectified,
            right_rectified=right_rectified,
            timestamp=float(timestamp),
            tracking_result=tracking_result,
            imu_rotation_prior=imu_rotation_prior,
        )

        self._set_frame_reference(result)
        self._update_map(left_rectified, tracking_result, result)
        self._append_result(result)
        return result

    def _update_map(self, left_rectified, tracking_result, result) -> None:
        if self.map is None or not result.success:
            return

        frame_index = len(self.results)
        if not self._should_create_keyframe(frame_index, tracking_result.active_ids):
            return

        descriptors, indices = self.tracker.describe(
            left_rectified,
            tracking_result.active_points,
        )
        if len(indices) == 0:
            return

        points_3d_rect = self.prev_points_3d_rect[indices]
        valid = np.isfinite(points_3d_rect).all(axis=1)
        valid[valid] = (
            (points_3d_rect[valid, 2] > 0.0)
            & (points_3d_rect[valid, 2] <= self.max_landmark_depth)
        )
        if not np.any(valid):
            return

        previous_keyframe = self.map.last_keyframe
        keyframe = self.map.add_keyframe(
            frame_index=frame_index,
            timestamp=result.timestamp,
            T_W_C=result.T_W_Cleft,
            points_2d=tracking_result.active_points[indices][valid],
            points_C=self._rectified_points_to_output_camera(points_3d_rect[valid]),
            descriptors=descriptors[valid],
            track_ids=tracking_result.active_ids[indices][valid],
        )
        self.map.prune_landmarks(self.local_matcher.keyframe_count)
        result.is_keyframe = True
        result.reference_keyframe_id = keyframe.id
        result.T_C_ref_C_frame = np.eye(4, dtype=np.float64)

        if previous_keyframe is not None:
            self.pose_graph.add_edge(
                previous_keyframe.id,
                keyframe.id,
                invert_transform(previous_keyframe.T_W_C) @ keyframe.T_W_C,
                "sequential",
            )
        if self.loop_enabled:
            self._try_loop_closure(keyframe, result)

    def _set_frame_reference(self, result: StereoPnPSLAMResult) -> None:
        if (
            result.reference_keyframe_id >= 0
            or self.map is None
            or self.map.last_keyframe is None
        ):
            return
        reference = self.map.last_keyframe
        result.reference_keyframe_id = reference.id
        result.T_C_ref_C_frame = (
            invert_transform(reference.T_W_C) @ result.T_W_Cleft
        )

    def _try_loop_closure(self, keyframe, result: StereoPnPSLAMResult) -> None:
        last_candidate_id = keyframe.id - self.loop_min_keyframe_gap
        if last_candidate_id < 0:
            return

        if keyframe.id < self.last_loop_keyframe_id + self.loop_min_interval:
            return

        num_candidates = last_candidate_id + 1

        current_pos = result.T_W_Cleft[:3, 3]
        distances = []
        for cid in range(num_candidates):
            cand_pos = self.map.keyframes[cid].T_W_C[:3, 3]
            distances.append((float(np.linalg.norm(cand_pos - current_pos)), cid))

        distances.sort()
        spatial_candidates = [cid for _, cid in distances[:self.loop_candidates_per_frame]]

        round_robin_count = self.loop_candidates_per_frame // 2
        start_idx = self.loop_candidate_check_index % num_candidates
        end_idx = start_idx + round_robin_count

        if end_idx <= num_candidates:
            rr_candidates = list(range(start_idx, end_idx))
            self.loop_candidate_check_index = end_idx
        else:
            rr_candidates = list(range(start_idx, num_candidates)) + list(range(0, end_idx - num_candidates))
            self.loop_candidate_check_index = end_idx - num_candidates

        candidate_ids = list(set(spatial_candidates + rr_candidates))

        recognition = self.place_recognizer.recognize(
            points_2d=keyframe.points_2d,
            descriptors=keyframe.descriptors,
            sparse_map=self.map,
            candidate_ids=candidate_ids,
            points_C=keyframe.points_C,
        )
        result.loop_candidate_count = len(recognition.candidates)
        if recognition.candidates:
            best = recognition.candidates[0]
            result.loop_candidate_id = best.keyframe_id
            result.loop_match_count = best.match_count
        if recognition.verification is None:
            return

        verification = recognition.verification
        solution = verification.solution
        candidate_id = verification.candidate.keyframe_id
        result.loop_candidate_id = candidate_id
        result.loop_match_count = verification.candidate.match_count
        result.loop_accepted = True
        result.loop_relative_scale = verification.relative_scale
        self.last_loop_keyframe_id = keyframe.id
        self.pose_graph.add_edge(
            candidate_id,
            keyframe.id,
            invert_transform(solution.T_Ck_Cprev),
            "loop",
            inlier_count=int(np.count_nonzero(solution.inlier_mask)),
            reprojection_error_median=solution.reprojection_error_median,
        )

        optimization = self.pose_graph.optimize(self.map.keyframes)
        result.graph_cost_before = optimization.cost_before
        result.graph_cost_after = optimization.cost_after
        self.last_graph_cost_before = optimization.cost_before
        self.last_graph_cost_after = optimization.cost_after
        if optimization.success:
            self._apply_graph_correction(optimization.poses, result)

    def _apply_graph_correction(self, poses, current_result) -> None:
        for keyframe, pose in zip(self.map.keyframes, poses):
            keyframe.T_W_C = pose
        self.map.update_world_positions()

        for result in self.results + [current_result]:
            if result.reference_keyframe_id < 0:
                continue
            reference = self.map.keyframes[result.reference_keyframe_id]
            result.T_W_Cleft = reference.T_W_C @ result.T_C_ref_C_frame

        self.T_W_Cleft = current_result.T_W_Cleft.copy()
        self.trajectory = Trajectory()
        for result in self.results:
            self.trajectory.append(
                result.timestamp,
                Pose.from_matrix(result.T_W_Cleft),
            )

    def _should_create_keyframe(
        self,
        frame_index: int,
        active_track_ids: np.ndarray,
    ) -> bool:
        last_keyframe = self.map.last_keyframe
        if last_keyframe is None:
            return True

        if frame_index - last_keyframe.frame_index < self.keyframe_min_frame_gap:
            return False

        if frame_index - last_keyframe.frame_index >= self.keyframe_max_frame_gap:
            return True

        T_Clast_C = invert_transform(last_keyframe.T_W_C) @ self.T_W_Cleft
        translation = float(np.linalg.norm(T_Clast_C[:3, 3]))
        rotation_deg = rotation_angle_deg(T_Clast_C[:3, :3])

        depths = self.prev_points_3d_rect[:, 2]
        depths = depths[np.isfinite(depths)]
        depths = depths[depths > 0.0]
        translation_ratio = (
            translation / float(np.median(depths)) if len(depths) else 0.0
        )
        tracked_ratio = self.map.tracked_landmark_count(active_track_ids) / max(
            1,
            len(active_track_ids),
        )

        return (
            translation_ratio >= self.keyframe_translation_depth_ratio
            or rotation_deg >= self.keyframe_rotation_deg
            or tracked_ratio < self.keyframe_tracked_landmark_ratio
        )

    def _rectified_points_to_output_camera(self, points_3d: np.ndarray) -> np.ndarray:
        """Convert rectified-left points to the camera frame used by T_W_Cleft."""
        return (self.R_C_Crect @ np.asarray(points_3d, dtype=np.float64).T).T

    def _estimate_current_pose(
        self,
        left_rectified: np.ndarray,
        right_rectified: np.ndarray,
        timestamp: float,
        tracking_result,
        imu_rotation_prior: np.ndarray = None,
    ) -> StereoPnPSLAMResult:
        map_correspondences = MapCorrespondences()
        if self.map is not None:
            map_correspondences = self.local_matcher.match(
                image=left_rectified,
                points=tracking_result.active_points,
                track_ids=tracking_result.active_ids,
                sparse_map=self.map,
                T_W_Crect=self.T_W_Cleft @ self.T_C_Crect,
            )

        map_solution = PnPSolution.failure(
            len(map_correspondences.object_points),
            "not enough map correspondences",
        )
        if len(map_correspondences.object_points) >= self.pnp.min_points:
            map_solution = self.pnp.solve(
                object_points=map_correspondences.object_points,
                image_points=map_correspondences.image_points,
                imu_rotation_prior=imu_rotation_prior,
                motion_converter=self._rectified_world_to_output_motion,
                refinement_T_Crect_object=(
                    self.T_Crect_C @ invert_transform(self.T_W_Cleft)
                ),
            )

        if map_solution.success:
            solution = map_solution
            pnp_image_points = map_correspondences.image_points
            pose_source = "map"
        else:
            pnp_object_points, pnp_image_points = self._make_pnp_correspondences(
                tracking_result
            )
            solution = PnPSolution.failure(
                len(pnp_object_points),
                "not enough PnP correspondences",
            )
            if len(pnp_object_points) >= self.pnp.min_points:
                solution = self.pnp.solve(
                    object_points=pnp_object_points,
                    image_points=pnp_image_points,
                    imu_rotation_prior=imu_rotation_prior,
                    motion_converter=self._rectified_motion_to_output_motion,
                )
            pose_source = "vo_fallback" if solution.success else "none"

        relocalized = False
        relocalization_keyframe_id = -1
        if not solution.success:
            relocalization = self._try_relocalization(
                left_rectified,
                tracking_result,
            )
            if relocalization is not None:
                verification, described_indices = relocalization
                solution = verification.solution
                current_indices = verification.candidate.current_indices
                pnp_image_points = tracking_result.active_points[
                    described_indices[current_indices]
                ].astype(np.float64)
                candidate = self.map.keyframes[
                    verification.candidate.keyframe_id
                ]
                relocalization_keyframe_id = candidate.id
                self.T_W_Cleft = candidate.T_W_C @ invert_transform(
                    solution.T_Ck_Cprev
                )
                inliers = solution.inlier_mask
                self.map.associate_tracks(
                    tracking_result.active_ids[
                        described_indices[current_indices]
                    ][inliers],
                    verification.landmark_ids[inliers],
                )
                pose_source = "relocalization"
                relocalized = True

        if solution.success and not relocalized:
            self.T_W_Cleft = self.T_W_Cleft @ invert_transform(
                solution.T_Ck_Cprev
            )
            if pose_source == "map":
                self.T_W_Cleft[:3, :3] = orthonormalize_rotation(
                    self.T_W_Cleft[:3, :3]
                )
        if (
            solution.success
            and self.map is not None
            and pose_source in ("map", "vo_fallback")
        ):
            association_mask = map_solution.inlier_mask
            if (
                pose_source == "vo_fallback"
                and len(map_correspondences.image_points)
            ):
                projected, visible = self.local_matcher.project(
                    points_W=map_correspondences.object_points,
                    T_W_Crect=self.T_W_Cleft @ self.T_C_Crect,
                )
                association_mask = visible & (
                    np.linalg.norm(
                        map_correspondences.image_points - projected,
                        axis=1,
                    )
                    <= self.local_matcher.reprojection_gate_px
                )
            self.map.associate_tracks(
                map_correspondences.track_ids[association_mask],
                map_correspondences.landmark_ids[association_mask],
            )

        points_3d, depth_count = self._compute_depth_for_active_points(
            left_rectified=left_rectified,
            right_rectified=right_rectified,
            active_points=tracking_result.active_points,
        )
        if pose_source == "vo_fallback" and self.map is not None:
            valid = np.isfinite(points_3d).all(axis=1)
            valid[valid] = (
                (points_3d[valid, 2] > 0.0)
                & (points_3d[valid, 2] <= self.max_landmark_depth)
            )
            self.map.update_landmark_positions(
                tracking_result.active_ids[valid],
                self._rectified_points_to_output_camera(points_3d[valid]),
                self.T_W_Cleft,
            )

        self.prev_points_3d_rect = points_3d
        pnp_inlier_count = int(np.count_nonzero(solution.inlier_mask))
        self.consecutive_failures = (
            0 if solution.success else self.consecutive_failures + 1
        )

        return StereoPnPSLAMResult(
            timestamp=timestamp,
            success=solution.success,
            initialized=True,
            T_W_Cleft=self.T_W_Cleft.copy(),
            track_count=tracking_result.track_count,
            pnp_point_count=len(pnp_image_points),
            pnp_inlier_count=pnp_inlier_count,
            pose_source=pose_source,
            map_point_count=len(map_correspondences.object_points),
            map_inlier_count=int(np.count_nonzero(map_solution.inlier_mask)),
            local_landmark_count=map_correspondences.local_landmark_count,
            map_descriptor_match_count=map_correspondences.descriptor_match_count,
            map_message=map_solution.message,
            new_feature_count=tracking_result.detected_count,
            reprojection_error_mean=solution.reprojection_error_mean,
            reprojection_error_median=solution.reprojection_error_median,
            pnp_rotation_step_deg=solution.pnp_rotation_step_deg,
            imu_rotation_step_deg=solution.imu_rotation_step_deg,
            pnp_imu_rotation_error_deg=solution.pnp_imu_rotation_error_deg,
            imu_rotation_consistent=solution.imu_rotation_consistent,
            imu_rejected=solution.imu_rejected,
            reinitialized=False,
            depth_count=depth_count,
            message=solution.message,
            tracked_points_curr=tracking_result.curr_points,
            pnp_points_curr=pnp_image_points.astype(np.float32),
            pnp_inlier_points_curr=pnp_image_points[solution.inlier_mask].astype(
                np.float32
            ),
            tracking_state="TRACKING" if solution.success else "LOST",
            reference_keyframe_id=relocalization_keyframe_id,
            T_C_ref_C_frame=(
                invert_transform(self.map.keyframes[relocalization_keyframe_id].T_W_C)
                @ self.T_W_Cleft
                if relocalization_keyframe_id >= 0
                else np.eye(4, dtype=np.float64)
            ),
            relocalized=relocalized,
        )

    def _try_relocalization(self, image, tracking_result):
        if (
            not self.relocalization_enabled
            or self.map is None
            or not self.map.keyframes
            or self.consecutive_failures + 1 < self.relocalization_failure_count
        ):
            return None

        descriptors, described_indices = self.tracker.describe(
            image,
            tracking_result.active_points,
        )
        if len(described_indices) == 0:
            return None

        recognition = self.place_recognizer.recognize(
            points_2d=tracking_result.active_points[described_indices],
            descriptors=descriptors,
            sparse_map=self.map,
            candidate_ids=range(len(self.map.keyframes)),
        )
        if recognition.verification is None:
            return None
        return recognition.verification, described_indices

    def _make_pnp_correspondences(self, tracking_result) -> tuple:
        if len(self.prev_points_3d_rect) == 0:
            return empty_points(3, dtype=np.float64), empty_points(2)

        status_mask = tracking_result.status_mask

        if len(status_mask) != len(self.prev_points_3d_rect):
            return empty_points(3, dtype=np.float64), empty_points(2)

        object_points = self.prev_points_3d_rect[status_mask]
        image_points = tracking_result.curr_points
        valid_mask = np.isfinite(object_points).all(axis=1)
        valid_mask &= object_points[:, 2] > 0.0

        return (
            object_points[valid_mask].astype(np.float64),
            image_points[valid_mask].astype(np.float64),
        )

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

    def _append_result(self, result: StereoPnPSLAMResult) -> None:
        if self.results:
            previous = self.results[-1]
            previous.tracked_points_curr = empty_points(2)
            previous.pnp_points_curr = empty_points(2)
            previous.pnp_inlier_points_curr = empty_points(2)
        self.trajectory.append(
            timestamp=result.timestamp,
            pose=Pose.from_matrix(result.T_W_Cleft),
        )
        self.results.append(result)

    def _rectified_motion_to_output_motion(self, T_Ckrect_Cprevrect) -> np.ndarray:
        return self.T_C_Crect @ T_Ckrect_Cprevrect @ self.T_Crect_C

    def _rectified_candidate_to_output_motion(self, T_Ckrect_Ccandidate) -> np.ndarray:
        return self.T_C_Crect @ T_Ckrect_Ccandidate

    def _rectified_world_to_output_motion(self, T_Ckrect_W) -> np.ndarray:
        T_Ck_Cprev = self.T_C_Crect @ T_Ckrect_W @ self.T_W_Cleft
        T_Ck_Cprev[:3, :3] = orthonormalize_rotation(T_Ck_Cprev[:3, :3])
        return T_Ck_Cprev
