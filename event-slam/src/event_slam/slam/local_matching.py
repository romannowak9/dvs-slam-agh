from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from event_slam.core.geometry import empty_points, invert_transform, transform_points


@dataclass
class MapCorrespondences:
    object_points: np.ndarray = field(
        default_factory=lambda: empty_points(3, dtype=np.float64)
    )
    image_points: np.ndarray = field(default_factory=lambda: empty_points(2))
    track_ids: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    landmark_ids: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    local_landmark_count: int = 0
    descriptor_match_count: int = 0


class LocalMapMatcher:
    """Match active image points with landmarks from recent keyframes."""

    def __init__(
        self,
        K: np.ndarray,
        descriptor_extractor,
        keyframe_count: int = 5,
        descriptor_ratio: float = 0.75,
        reprojection_gate_px: float = 15.0,
    ) -> None:
        self.K = np.asarray(K, dtype=np.float64)
        self.descriptor_extractor = descriptor_extractor
        self.keyframe_count = int(keyframe_count)
        self.descriptor_ratio = float(descriptor_ratio)
        self.reprojection_gate_px = float(reprojection_gate_px)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def match(
        self,
        image: np.ndarray,
        points: np.ndarray,
        track_ids: np.ndarray,
        sparse_map,
        T_W_Crect: np.ndarray,
    ) -> MapCorrespondences:
        local_landmarks = sparse_map.local_landmarks(self.keyframe_count)
        local_by_id = {landmark.id: landmark for landmark in local_landmarks}
        point_indices = []
        landmark_ids = []
        used_landmarks = set()

        for point_index, track_id_value in enumerate(track_ids):
            landmark_id = sparse_map.track_to_landmark.get(int(track_id_value))
            if landmark_id in local_by_id and landmark_id not in used_landmarks:
                point_indices.append(point_index)
                landmark_ids.append(landmark_id)
                used_landmarks.add(landmark_id)

        used_points = set(point_indices)
        descriptor_matches = self._match_descriptors(
            image,
            points,
            used_points,
            used_landmarks,
            local_landmarks,
            T_W_Crect,
        )
        for point_index, landmark_id in descriptor_matches:
            point_indices.append(point_index)
            landmark_ids.append(landmark_id)

        if not point_indices:
            return MapCorrespondences(local_landmark_count=len(local_landmarks))

        point_indices = np.asarray(point_indices, dtype=np.int64)
        landmark_ids = np.asarray(landmark_ids, dtype=np.int64)
        return MapCorrespondences(
            object_points=np.asarray(
                [sparse_map.landmarks[int(item)].position_W for item in landmark_ids],
                dtype=np.float64,
            ),
            image_points=points[point_indices].astype(np.float64),
            track_ids=track_ids[point_indices].astype(np.int64),
            landmark_ids=landmark_ids,
            local_landmark_count=len(local_landmarks),
            descriptor_match_count=len(descriptor_matches),
        )

    def project(
        self,
        points_W: np.ndarray,
        T_W_Crect: np.ndarray,
    ) -> tuple:
        points_Crect = transform_points(invert_transform(T_W_Crect), points_W)
        visible = np.isfinite(points_Crect).all(axis=1)
        visible[visible] = points_Crect[visible, 2] > 0.0
        projected = np.full((len(points_W), 2), np.nan, dtype=np.float64)
        image_h = (self.K @ points_Crect[visible].T).T
        projected[visible] = image_h[:, :2] / image_h[:, 2:3]
        return projected, visible

    def _match_descriptors(
        self,
        image,
        points,
        used_points,
        used_landmarks,
        local_landmarks,
        T_W_Crect,
    ) -> list:
        train_landmarks = [
            landmark
            for landmark in local_landmarks
            if landmark.id not in used_landmarks
        ]
        if len(train_landmarks) < 2:
            return []

        descriptors, described_indices = self.descriptor_extractor(image, points)
        query_mask = np.asarray(
            [int(index) not in used_points for index in described_indices]
        )
        if not np.any(query_mask):
            return []

        projected, visible = self.project(
            np.asarray([landmark.position_W for landmark in train_landmarks]),
            T_W_Crect,
        )
        train_landmarks = [
            landmark for landmark, keep in zip(train_landmarks, visible) if keep
        ]
        projected = projected[visible]
        if len(train_landmarks) < 2:
            return []

        query_descriptors = descriptors[query_mask]
        query_indices = described_indices[query_mask]
        train_descriptors = np.asarray(
            [landmark.descriptor for landmark in train_landmarks],
            dtype=np.uint8,
        )
        candidates = []
        for matches in self.matcher.knnMatch(
            query_descriptors,
            train_descriptors,
            k=2,
        ):
            if len(matches) < 2:
                continue
            best, second = matches
            point_index = int(query_indices[best.queryIdx])
            if (
                best.distance < self.descriptor_ratio * second.distance
                and np.linalg.norm(points[point_index] - projected[best.trainIdx])
                <= self.reprojection_gate_px
            ):
                candidates.append(
                    (
                        best.distance,
                        point_index,
                        train_landmarks[best.trainIdx].id,
                    )
                )

        matches = []
        for _, point_index, landmark_id in sorted(candidates):
            if point_index in used_points or landmark_id in used_landmarks:
                continue
            used_points.add(point_index)
            used_landmarks.add(landmark_id)
            matches.append((point_index, landmark_id))
        return matches
