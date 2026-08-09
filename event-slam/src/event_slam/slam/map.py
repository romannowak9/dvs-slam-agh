from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from event_slam.core.geometry import invert_transform, transform_points


@dataclass
class Keyframe:
    id: int
    frame_index: int
    timestamp: float
    T_W_C: np.ndarray
    points_2d: np.ndarray
    points_C: np.ndarray
    descriptors: np.ndarray
    landmark_ids: np.ndarray

    @property
    def point_count(self) -> int:
        return len(self.landmark_ids)


@dataclass
class Landmark:
    id: int
    anchor_keyframe_id: int
    position_C_anchor: np.ndarray
    position_W: np.ndarray
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

    def local_landmarks(self, keyframe_count: int = 5) -> list:
        """Return unique mapped landmarks seen by the latest keyframes."""
        landmark_ids = dict.fromkeys(
            int(landmark_id)
            for keyframe in self.keyframes[-int(keyframe_count) :]
            for landmark_id in keyframe.landmark_ids
            if int(landmark_id) in self.landmarks
        )
        return [self.landmarks[landmark_id] for landmark_id in landmark_ids]

    def associate_tracks(
        self,
        track_ids: np.ndarray,
        landmark_ids: np.ndarray,
    ) -> None:
        for track_id, landmark_id in zip(track_ids, landmark_ids):
            self.track_to_landmark[int(track_id)] = int(landmark_id)

    def prune_landmarks(self, recent_keyframes: int = 5) -> int:
        """Remove unconfirmed landmarks outside the recent keyframe window."""
        if len(self.keyframes) <= recent_keyframes:
            return 0

        oldest_recent_id = self.keyframes[-int(recent_keyframes)].id
        removed_ids = {
            landmark_id
            for landmark_id, landmark in self.landmarks.items()
            if landmark.observation_count < 2
            and landmark.anchor_keyframe_id < oldest_recent_id
        }
        if not removed_ids:
            return 0

        for landmark_id in removed_ids:
            del self.landmarks[landmark_id]

        self.track_to_landmark = {
            track_id: landmark_id
            for track_id, landmark_id in self.track_to_landmark.items()
            if landmark_id not in removed_ids
        }
        for keyframe in self.keyframes:
            keep = np.asarray(
                [
                    int(landmark_id) not in removed_ids
                    for landmark_id in keyframe.landmark_ids
                ]
            )
            keyframe.points_2d = keyframe.points_2d[keep]
            keyframe.points_C = keyframe.points_C[keep]
            keyframe.descriptors = keyframe.descriptors[keep]
            keyframe.landmark_ids = keyframe.landmark_ids[keep]

        return len(removed_ids)

    def update_world_positions(self) -> None:
        """Move anchored landmarks with their optimized keyframes."""
        for landmark in self.landmarks.values():
            anchor = self.keyframes[landmark.anchor_keyframe_id]
            landmark.position_W = transform_points(
                anchor.T_W_C,
                landmark.position_C_anchor,
            )

    def update_landmark_positions(
        self,
        track_ids: np.ndarray,
        points_C: np.ndarray,
        T_W_C: np.ndarray,
    ) -> None:
        """Refresh visible landmarks after a successful VO fallback."""
        positions_W = transform_points(T_W_C, points_C)
        for track_id, position_W in zip(track_ids, positions_W):
            landmark_id = self.track_to_landmark.get(int(track_id))
            if landmark_id not in self.landmarks:
                continue
            landmark = self.landmarks[landmark_id]
            landmark.position_W = position_W.copy()
            anchor = self.keyframes[landmark.anchor_keyframe_id]
            landmark.position_C_anchor = transform_points(
                invert_transform(anchor.T_W_C),
                position_W.reshape(1, 3),
            )[0]

    def add_keyframe(
        self,
        frame_index: int,
        timestamp: float,
        T_W_C: np.ndarray,
        points_2d: np.ndarray,
        points_C: np.ndarray,
        descriptors: np.ndarray,
        track_ids: np.ndarray,
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
                observation_count = landmark.observation_count + 1
                landmark.position_W += (
                    positions_W[index] - landmark.position_W
                ) / observation_count
                anchor = self.keyframes[landmark.anchor_keyframe_id]
                landmark.position_C_anchor = transform_points(
                    invert_transform(anchor.T_W_C),
                    landmark.position_W.reshape(1, 3),
                )[0]
                landmark.descriptor = descriptors[index].copy()
                landmark.observation_count = observation_count
                landmark.last_seen_keyframe_id = keyframe_id

            landmark_ids[index] = landmark_id

        keyframe = Keyframe(
            id=keyframe_id,
            frame_index=int(frame_index),
            timestamp=float(timestamp),
            T_W_C=np.asarray(T_W_C, dtype=np.float64).copy(),
            points_2d=np.asarray(points_2d, dtype=np.float32).copy(),
            points_C=np.asarray(points_C, dtype=np.float64).copy(),
            descriptors=np.asarray(descriptors, dtype=np.uint8).copy(),
            landmark_ids=landmark_ids,
        )
        self.keyframes.append(keyframe)
        return keyframe
