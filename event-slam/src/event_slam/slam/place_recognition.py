from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from event_slam.core.geometry import invert_transform
from event_slam.vo.pnp import PnPSolution, PnPSolver


_CANDIDATE_POOL_SIZE = 15


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
    solution: PnPSolution
    descriptor_match_count: int


@dataclass
class PlaceRecognitionResult:
    candidates: list
    verification: Optional[PlaceVerification] = None


@dataclass
class PlaceObservation:
    points_2d: np.ndarray
    points_C: np.ndarray
    descriptors: np.ndarray
    signature: np.ndarray


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
        pnp_reprojection_error: float = 3.0,
        pnp_confidence: float = 0.999,
        pnp_iterations: int = 100,
        max_features: int = 1500,
        max_depth: float = np.inf,
    ) -> None:
        self.descriptor_ratio = float(descriptor_ratio)
        self.min_matches = int(min_matches)
        self.max_candidates = int(max_candidates)
        self.R_Crect_C = np.asarray(R_Crect_C, dtype=np.float64)
        self.motion_converter = motion_converter
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.orb = cv2.ORB_create(nfeatures=int(max_features))
        self.max_depth = float(max_depth)
        self.pnp = PnPSolver(
            K=K,
            min_points=min_inliers,
            min_inliers=min_inliers,
            min_inlier_ratio=min_inlier_ratio,
            max_reprojection_median=max_reprojection_median,
            max_translation_step=np.inf,
            max_rotation_step_deg=180.0,
            reprojection_error=pnp_reprojection_error,
            confidence=pnp_confidence,
            iterations=pnp_iterations,
        )

    def observe(self, left_image, right_image, depth_estimator) -> PlaceObservation:
        """Build a polarity-independent place observation with stereo depth."""
        left = self._activity_image(left_image)
        right = self._activity_image(right_image)
        keypoints, descriptors = self.orb.detectAndCompute(left, None)
        if descriptors is None:
            return PlaceObservation(
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 32), dtype=np.uint8),
                self._signature(left),
            )

        points_2d = cv2.KeyPoint_convert(keypoints).reshape(-1, 2).astype(
            np.float32
        )
        depth = depth_estimator.estimate(left, right, points_2d)
        points_C = np.full((len(points_2d), 3), np.nan, dtype=np.float64)
        depth_mask = depth.points_3d_left_camera[:, 2] <= self.max_depth
        point_indices = np.flatnonzero(depth.valid_mask)[depth_mask]
        points_C[point_indices] = (
            self.R_Crect_C.T
            @ depth.points_3d_left_camera[depth_mask].T
        ).T
        return PlaceObservation(
            points_2d=points_2d,
            points_C=points_C,
            descriptors=descriptors,
            signature=self._signature(left),
        )

    @staticmethod
    def _activity_image(image):
        activity = 2 * np.abs(np.asarray(image, dtype=np.int16) - 127)
        return cv2.medianBlur(np.minimum(activity, 255).astype(np.uint8), 3)

    @staticmethod
    def _signature(image):
        signature = cv2.resize(image, (40, 24), interpolation=cv2.INTER_AREA)
        signature = signature.astype(np.float32).reshape(-1)
        signature -= np.mean(signature)
        return signature / max(float(np.linalg.norm(signature)), 1e-6)

    def recognize(
        self,
        points_2d: np.ndarray,
        descriptors: np.ndarray,
        sparse_map,
        candidate_ids,
        points_C: np.ndarray,
        signature: np.ndarray,
    ) -> PlaceRecognitionResult:
        candidate_ids = self._rank_candidates(
            signature,
            sparse_map,
            candidate_ids,
        )
        candidates = self.find_candidates(
            descriptors,
            sparse_map,
            candidate_ids,
        )
        best_verification = None
        for candidate in candidates[: self.max_candidates]:
            verification = self.verify(
                points_2d,
                candidate,
                sparse_map,
                points_C,
            )
            if verification.solution.success:
                return PlaceRecognitionResult(candidates, verification)
            if (
                best_verification is None
                or np.count_nonzero(verification.solution.inlier_mask)
                > np.count_nonzero(best_verification.solution.inlier_mask)
            ):
                best_verification = verification
        return PlaceRecognitionResult(candidates, best_verification)

    @staticmethod
    def _rank_candidates(signature, sparse_map, candidate_ids) -> list:
        """Rank the complete history by a cheap normalized activity signature."""
        ranked = (
            (
                float(signature @ sparse_map.keyframes[keyframe_id].place_signature),
                keyframe_id,
            )
            for keyframe_id in candidate_ids
        )
        return [
            keyframe_id
            for _, keyframe_id in sorted(ranked, reverse=True)[
                :_CANDIDATE_POOL_SIZE
            ]
        ]

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
            if len(keyframe.place_descriptors) < 2:
                continue

            matches = []
            for pair in self.matcher.knnMatch(
                descriptors, keyframe.place_descriptors, k=2
            ):
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
            score = len(unique) / min(
                len(descriptors), len(keyframe.place_descriptors)
            )
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
        current_points_C: np.ndarray,
    ) -> PlaceVerification:
        keyframe = sparse_map.keyframes[candidate.keyframe_id]
        candidate_points_C = keyframe.place_points_C[candidate.candidate_indices]
        current_points_C = np.asarray(current_points_C, dtype=np.float64)
        solutions = [
            self._verify_direction(
                candidate,
                candidate_points_C,
                current_points_2d,
                reverse=False,
            ),
            self._verify_direction(
                candidate,
                current_points_C[candidate.current_indices],
                keyframe.place_points_2d,
                reverse=True,
            ),
        ]
        return max(
            solutions,
            key=lambda item: (
                item.solution.success,
                int(np.count_nonzero(item.solution.inlier_mask)),
            ),
        )

    def _verify_direction(
        self,
        candidate,
        object_points,
        image_points,
        reverse,
    ) -> PlaceVerification:
        valid = np.isfinite(object_points).all(axis=1)
        filtered = PlaceCandidate(
            keyframe_id=candidate.keyframe_id,
            score=candidate.score,
            current_indices=candidate.current_indices[valid],
            candidate_indices=candidate.candidate_indices[valid],
        )
        solution = self.pnp.solve(
            object_points=object_points[valid],
            image_points=(
                image_points[candidate.candidate_indices[valid]]
                if reverse
                else image_points[candidate.current_indices[valid]]
            ),
            motion_converter=self.motion_converter,
        )
        if reverse and solution.success:
            solution.T_Ck_Cprev = invert_transform(solution.T_Ck_Cprev)
        return PlaceVerification(
            candidate=filtered,
            solution=solution,
            descriptor_match_count=candidate.match_count,
        )
