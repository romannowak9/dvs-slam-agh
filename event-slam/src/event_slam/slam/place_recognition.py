from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

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


@dataclass
class PlaceRecognitionResult:
    candidates: list
    verification: Optional[PlaceVerification] = None


class PlaceRecognizer:
    """ORB place candidates and shared geometric verification for SLAM."""

    def __init__(
        self,
        K: np.ndarray,
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
    ) -> None:
        self.descriptor_ratio = float(descriptor_ratio)
        self.min_matches = int(min_matches)
        self.max_candidates = int(max_candidates)
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
            valid = np.asarray(
                [
                    int(landmark_id) in sparse_map.landmarks
                    and np.isfinite(point_C).all()
                    for landmark_id, point_C in zip(
                        keyframe.landmark_ids,
                        keyframe.points_C,
                    )
                ]
            )
            train_indices = np.flatnonzero(valid)
            if len(train_indices) < 2:
                continue

            train_descriptors = keyframe.descriptors[train_indices]
            matches = []
            for pair in self.matcher.knnMatch(
                descriptors,
                train_descriptors,
                k=2,
            ):
                if len(pair) < 2:
                    continue
                best, second = pair
                if best.distance < self.descriptor_ratio * second.distance:
                    matches.append(
                        (
                            best.distance,
                            int(best.queryIdx),
                            int(train_indices[best.trainIdx]),
                        )
                    )

            unique = []
            used_train = set()
            for _, query_index, train_index in sorted(matches):
                if train_index in used_train:
                    continue
                used_train.add(train_index)
                unique.append((query_index, train_index))

            if len(unique) < self.min_matches:
                continue

            current_indices, candidate_indices = zip(*unique)
            score = len(unique) / min(len(descriptors), len(train_descriptors))
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
    ) -> PlaceVerification:
        keyframe = sparse_map.keyframes[candidate.keyframe_id]
        object_points = keyframe.points_C[candidate.candidate_indices]
        image_points = current_points_2d[candidate.current_indices]
        solution = self.pnp.solve(
            object_points=object_points,
            image_points=image_points,
            motion_converter=self.motion_converter,
        )
        return PlaceVerification(
            candidate=candidate,
            landmark_ids=keyframe.landmark_ids[candidate.candidate_indices],
            solution=solution,
        )
