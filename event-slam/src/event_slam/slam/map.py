from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from event_slam.core.geometry import transform_points


@dataclass
class Keyframe:
    id: int
    frame_index: int
    timestamp: float
    T_W_C: np.ndarray  # Camera pose in world coordinates (4x4 transformation matrix)
    points_2d: np.ndarray  # 2D points in the image plane
    descriptors: np.ndarray  # Descriptors for each point
    landmark_ids: np.ndarray  # IDs of landmarks associated with each point

    @property
    def point_count(self) -> int:
        return len(self.landmark_ids)


@dataclass
class Landmark:
    id: int
    anchor_keyframe_id: int
    position_C_anchor: np.ndarray  # Position of the landmark in the anchor keyframe's coordinate system
    position_W: np.ndarray  # Position of the landmark in world coordinates
    descriptor: np.ndarray
    observation_count: int
    last_seen_keyframe_id: int


class SparseMap:
    """Small keyframe and landmark map with track-to-landmark associations."""

    def __init__(self) -> None:
        self.keyframes = []
        self.landmarks = {}
        self.track_to_landmark = {}
        self.next_landmark_id = 0

    @property
    def last_keyframe(self):
        return self.keyframes[-1] if self.keyframes else None

    def tracked_landmark_count(self, track_ids: np.ndarray) -> int:
        return sum(int(track_id) in self.track_to_landmark for track_id in track_ids)

    def add_keyframe(
        self,
        frame_index: int,
        timestamp: float,
        T_W_C: np.ndarray,  # Camera pose in world coordinates (4x4 transformation matrix)
        points_2d: np.ndarray,  # 2D points in the image plane
        points_C: np.ndarray,  # 3D points in the camera coordinate system
        descriptors: np.ndarray,  # Descriptors for each point
        track_ids: np.ndarray,  # IDs of tracks associated with each point
    ) -> Keyframe:
        keyframe_id = len(self.keyframes)
        positions_W = transform_points(T_W_C, points_C)
        landmark_ids = np.empty(len(track_ids), dtype=np.int64)

        for index, track_id_value in enumerate(track_ids):
            track_id = int(track_id_value)
            landmark_id = self.track_to_landmark.get(track_id)

            if landmark_id is None:
                landmark_id = self.next_landmark_id
                self.next_landmark_id += 1
                self.landmarks[landmark_id] = Landmark(
                    id=landmark_id,
                    anchor_keyframe_id=keyframe_id,
                    position_C_anchor=points_C[index].copy(),
                    position_W=positions_W[index].copy(),
                    descriptor=descriptors[index].copy(),
                    observation_count=1,
                    last_seen_keyframe_id=keyframe_id,
                )
                self.track_to_landmark[track_id] = landmark_id
            else:
                landmark = self.landmarks[landmark_id]
                landmark.observation_count += 1
                landmark.last_seen_keyframe_id = keyframe_id

            landmark_ids[index] = landmark_id

        keyframe = Keyframe(
            id=keyframe_id,
            frame_index=int(frame_index),
            timestamp=float(timestamp),
            T_W_C=np.asarray(T_W_C, dtype=np.float64).copy(),
            points_2d=np.asarray(points_2d, dtype=np.float32).copy(),
            descriptors=np.asarray(descriptors, dtype=np.uint8).copy(),
            landmark_ids=landmark_ids,
        )
        self.keyframes.append(keyframe)
        return keyframe
