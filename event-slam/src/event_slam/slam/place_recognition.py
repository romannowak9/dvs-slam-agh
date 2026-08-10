from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from event_slam.core.geometry import invert_transform, transform_points
from event_slam.vo.pnp import PnPSolution, PnPSolver


@dataclass
class PlaceCandidate:
    keyframe_id: int
    score: float
    current_indices: np.ndarray
    candidate_indices: np.ndarray

    @property
    def match_count(self) -> int:
        return len(self.current_indices)


@dataclass
class PlaceVerification:
    candidate: PlaceCandidate
    landmark_ids: np.ndarray
    solution: PnPSolution
    relative_scale: float = 1.0


@dataclass
class PlaceRecognitionResult:
    candidates: list
    verification: Optional[PlaceVerification] = None


class PlaceRecognizer:
    """ORB place candidates and shared geometric verification for SLAM."""

    def __init__(
        self,
        K: np.ndarray,
        R_Crect_C: np.ndarray,
        motion_converter,
        descriptor_ratio: float = 0.75,
        min_matches: int = 30,
        max_candidates: int = 3,
        min_inliers: int = 20,
        min_inlier_ratio: float = 0.3,
        max_reprojection_median: float = 3.0,
        min_scale_error_reduction: float = 0.2,
        pnp_reprojection_error: float = 3.0,
        pnp_confidence: float = 0.999,
        pnp_iterations: int = 100,
    ) -> None:
        self.descriptor_ratio = float(descriptor_ratio)
        self.min_matches = int(min_matches)
        self.max_candidates = int(max_candidates)
        self.K = np.asarray(K, dtype=np.float64)
        self.R_Crect_C = np.asarray(R_Crect_C, dtype=np.float64)
        self.min_scale_error_reduction = float(min_scale_error_reduction)
        self.motion_converter = motion_converter
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.pnp = PnPSolver(
            K=K,
            min_points=min_matches,
            min_inliers=min_inliers,
            min_inlier_ratio=min_inlier_ratio,
            max_reprojection_median=max_reprojection_median,
            max_translation_step=np.inf,
            max_rotation_step_deg=180.0,
            reprojection_error=pnp_reprojection_error,
            confidence=pnp_confidence,
            iterations=pnp_iterations,
        )

    def recognize(
        self,
        points_2d: np.ndarray,
        descriptors: np.ndarray,
        sparse_map,
        candidate_ids,
        points_C: np.ndarray = None,
    ) -> PlaceRecognitionResult:
        candidates = self.find_candidates(
            descriptors,
            sparse_map,
            candidate_ids,
        )
        for candidate in candidates[: self.max_candidates]:
            verification = self.verify(
                points_2d,
                candidate,
                sparse_map,
                points_C,
            )
            if verification.solution.success:
                return PlaceRecognitionResult(candidates, verification)
        return PlaceRecognitionResult(candidates)

    def find_candidates(
        self,
        descriptors: np.ndarray,
        sparse_map,
        candidate_ids,
    ) -> list:
        descriptors = np.asarray(descriptors, dtype=np.uint8)
        if len(descriptors) < 2:
            return []

        candidates = []
        for keyframe_id in candidate_ids:
            keyframe = sparse_map.keyframes[int(keyframe_id)]
            if len(keyframe.descriptors) < 2:
                continue

            matches = []
            for pair in self.matcher.knnMatch(descriptors, keyframe.descriptors, k=2):
                if len(pair) < 2:
                    continue
                best, second = pair
                if best.distance < self.descriptor_ratio * second.distance:
                    matches.append(best)

            used_train = set()
            unique = []
            for match in sorted(matches, key=lambda item: item.distance):
                if match.trainIdx not in used_train:
                    used_train.add(match.trainIdx)
                    unique.append((match.queryIdx, match.trainIdx))

            if len(unique) < self.min_matches:
                continue

            current_indices, candidate_indices = zip(*unique)
            score = len(unique) / min(len(descriptors), len(keyframe.descriptors))
            candidates.append(
                PlaceCandidate(
                    keyframe_id=keyframe.id,
                    score=float(score),
                    current_indices=np.asarray(current_indices, dtype=np.int64),
                    candidate_indices=np.asarray(candidate_indices, dtype=np.int64),
                )
            )

        return sorted(candidates, key=lambda item: item.score, reverse=True)

    def verify(
        self,
        current_points_2d: np.ndarray,
        candidate: PlaceCandidate,
        sparse_map,
        current_points_C: np.ndarray = None,
    ) -> PlaceVerification:
        keyframe = sparse_map.keyframes[candidate.keyframe_id]
        object_points = keyframe.points_C[candidate.candidate_indices]
        image_points = current_points_2d[candidate.current_indices]
        solution = self.pnp.solve(
            object_points=object_points,
            image_points=image_points,
            motion_converter=self.motion_converter,
        )
        relative_scale = 1.0
        if solution.success and current_points_C is not None:
            relative_scale = self._refine_stereo_motion(
                solution,
                candidate,
                keyframe,
                current_points_2d,
                current_points_C,
            )
        return PlaceVerification(
            candidate=candidate,
            landmark_ids=keyframe.landmark_ids[candidate.candidate_indices],
            solution=solution,
            relative_scale=relative_scale,
        )

    def _refine_stereo_motion(
        self,
        solution,
        candidate,
        candidate_keyframe,
        current_points_2d,
        current_points_C,
    ) -> float:
        """Express the loop translation in the current stereo depth scale."""
        inliers = solution.inlier_mask
        candidate_indices = candidate.candidate_indices[inliers]
        current_indices = candidate.current_indices[inliers]
        candidate_3d = candidate_keyframe.points_C[candidate_indices]
        current_3d = current_points_C[current_indices]
        candidate_2d = candidate_keyframe.points_2d[candidate_indices]
        current_2d = current_points_2d[current_indices]

        first, second = np.triu_indices(len(candidate_3d), k=1)
        candidate_distances = np.linalg.norm(
            candidate_3d[first] - candidate_3d[second], axis=1
        )
        current_distances = np.linalg.norm(
            current_3d[first] - current_3d[second], axis=1
        )
        valid = candidate_distances > 1e-6
        if not np.any(valid):
            solution.success = False
            solution.message = "stereo refinement failed"
            return np.nan

        scale = float(
            np.median(current_distances[valid] / candidate_distances[valid])
        )
        T_current_candidate = solution.T_Ck_Cprev
        T_scaled = T_current_candidate.copy()
        T_scaled[:3, 3] *= scale
        forward_errors = self._reprojection_errors(
            T_current_candidate, candidate_3d, current_2d
        )
        scaled_errors = self._reprojection_errors(
            invert_transform(T_scaled), current_3d, candidate_2d
        )
        unit_scale_errors = self._reprojection_errors(
            invert_transform(T_current_candidate), current_3d, candidate_2d
        )
        if np.median(scaled_errors) >= (
            1.0 - self.min_scale_error_reduction
        ) * np.median(unit_scale_errors):
            scale = 1.0
            scaled_errors = unit_scale_errors

        errors = np.concatenate((forward_errors, scaled_errors))
        solution.T_Ck_Cprev[:3, 3] *= scale
        solution.reprojection_error_mean = float(np.mean(errors))
        solution.reprojection_error_median = float(np.median(errors))
        solution.success = (
            np.isfinite(scale)
            and 0.5 <= scale <= 2.0
            and solution.reprojection_error_median
            <= self.pnp.max_reprojection_median
        )
        solution.message = "ok" if solution.success else "stereo refinement failed"
        return scale

    def _reprojection_errors(
        self,
        T_target_source: np.ndarray,
        points_source: np.ndarray,
        pixels_target: np.ndarray,
    ) -> np.ndarray:
        projected = self._project(transform_points(T_target_source, points_source))
        return np.linalg.norm(projected - pixels_target, axis=1)

    def _project(self, points_C: np.ndarray) -> np.ndarray:
        points_rect = (self.R_Crect_C @ points_C.T).T
        pixels = (self.K @ points_rect.T).T
        return pixels[:, :2] / np.maximum(pixels[:, 2:3], 1e-9)
