from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from event_slam.core.geometry import Pose, interpolate_pose


@dataclass()
class PoseSample:
    """
    A single trajectory sample.

    The pose follows the project convention T_W_C.
    """

    timestamp: float
    pose: Pose

    def __post_init__(self) -> None:
        self.timestamp = float(self.timestamp)


@dataclass
class Trajectory:
    """
    A simple camera-pose trajectory.

    Stores consecutive T_W_Cleft poses.
    """

    samples: list[PoseSample] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def is_empty(self) -> bool:
        return len(self.samples) == 0

    @property
    def timestamps(self) -> np.ndarray:
        return np.array([sample.timestamp for sample in self.samples], dtype=np.float64)

    @property
    def positions(self) -> np.ndarray:
        if self.is_empty:
            return np.empty((0, 3), dtype=np.float64)

        return np.vstack([sample.pose.t for sample in self.samples]).astype(np.float64)

    @property
    def quaternions_xyzw(self) -> np.ndarray:
        if self.is_empty:
            return np.empty((0, 4), dtype=np.float64)

        return np.vstack(
            [sample.pose.as_quat_xyzw() for sample in self.samples]
        ).astype(np.float64)

    def append(self, timestamp: float, pose: Pose) -> None:
        timestamp = float(timestamp)

        if self.samples and timestamp < self.samples[-1].timestamp:
            raise ValueError(
                "Trajectory timestamps must be non-decreasing. "
                f"Last={self.samples[-1].timestamp}, new={timestamp}"
            )

        self.samples.append(PoseSample(timestamp=timestamp, pose=pose.copy()))

    def extend(self, samples: Iterable[PoseSample]) -> None:
        for sample in samples:
            self.append(sample.timestamp, sample.pose)

    def first(self) -> PoseSample:
        if self.is_empty:
            raise ValueError("Trajectory is empty")

        return self.samples[0]

    def last(self) -> PoseSample:
        if self.is_empty:
            raise ValueError("Trajectory is empty")

        return self.samples[-1]

    def interpolate(self, timestamp: float, *, clamp: bool = False) -> Pose:
        """
        Interpolate the trajectory at a given timestamp.

        Parameters:
        timestamp:
            Timestamp in seconds.
        clamp:
            If True, timestamps outside the trajectory range are clamped to the
            first or last pose. If False, an out-of-range timestamp raises ValueError.
        """
        return self._interpolate(float(timestamp), self.timestamps, clamp)

    def _interpolate(
        self,
        timestamp: float,
        timestamps: np.ndarray,
        clamp: bool,
    ) -> Pose:
        if self.is_empty:
            raise ValueError("Cannot interpolate an empty trajectory")

        if timestamp < timestamps[0]:
            if clamp:
                return self.samples[0].pose.copy()

            raise ValueError(
                f"Timestamp {timestamp} is before trajectory start {timestamps[0]}"
            )

        if timestamp > timestamps[-1]:
            if clamp:
                return self.samples[-1].pose.copy()

            raise ValueError(
                f"Timestamp {timestamp} is after trajectory end {timestamps[-1]}"
            )

        idx = bisect_left(timestamps, timestamp)

        if idx == 0:
            return self.samples[0].pose.copy()

        if idx < len(timestamps) and np.isclose(timestamps[idx], timestamp):
            return self.samples[idx].pose.copy()

        if idx >= len(timestamps):
            return self.samples[-1].pose.copy()

        sample0 = self.samples[idx - 1]
        sample1 = self.samples[idx]

        dt = sample1.timestamp - sample0.timestamp
        if dt <= 0.0:
            return sample0.pose.copy()

        alpha = (timestamp - sample0.timestamp) / dt
        return interpolate_pose(sample0.pose, sample1.pose, alpha)

    def interpolate_many(
        self,
        timestamps: np.ndarray,
        *,
        clamp: bool = False,
    ) -> Trajectory:
        """
        Interpolate the trajectory at multiple timestamps.
        """
        output = Trajectory()
        source_timestamps = self.timestamps

        for timestamp in np.asarray(timestamps, dtype=np.float64).reshape(-1):
            output.append(
                float(timestamp),
                self._interpolate(float(timestamp), source_timestamps, clamp),
            )

        return output

    def as_tum_array(self) -> np.ndarray:
        """
        Return the trajectory in TUM-style format:

            timestamp tx ty tz qx qy qz qw

        The quaternion order is xyzw.
        """
        if self.is_empty:
            return np.empty((0, 8), dtype=np.float64)

        ts = self.timestamps.reshape(-1, 1)
        positions = self.positions
        quaternions = self.quaternions_xyzw

        return np.hstack((ts, positions, quaternions)).astype(np.float64)

    @classmethod
    def from_tum_array(cls, data: np.ndarray) -> Trajectory:
        """
        Build a trajectory from an array with columns:

            timestamp tx ty tz qx qy qz qw
        """
        arr = np.asarray(data, dtype=np.float64)

        if arr.ndim != 2 or arr.shape[1] != 8:
            raise ValueError(
                f"TUM trajectory array must have shape (N, 8), got {arr.shape}"
            )

        trajectory = cls()

        for row in arr:
            timestamp = float(row[0])
            t = row[1:4]
            q_xyzw = row[4:8]
            pose = Pose.from_quat_xyzw(t=t, q_xyzw=q_xyzw)
            trajectory.append(timestamp, pose)

        return trajectory
