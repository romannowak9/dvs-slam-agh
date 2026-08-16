from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from event_slam.core.geometry import invert_transform
from event_slam.core.imu import rotation_angle_deg
from event_slam.core.trajectory import Trajectory


@dataclass
class PoseMetrics:
    position_mean: float
    position_rmse: float
    position_median: float
    position_max: float
    rotation_mean_deg: float
    rotation_rmse_deg: float
    rotation_median_deg: float
    rotation_max_deg: float
    rpe_pair_count: int
    rpe_delta_median_s: float
    rpe_translation_rmse: float
    rpe_rotation_rmse_deg: float


def match_timestamps(
    estimate_timestamps: np.ndarray,
    gt_timestamps: np.ndarray,
    tolerance: float,
) -> tuple:
    """Return monotonic one-to-one timestamp correspondences."""
    estimate_indices = []
    gt_indices = []
    estimate_index = gt_index = 0
    while estimate_index < len(estimate_timestamps) and gt_index < len(gt_timestamps):
        difference = estimate_timestamps[estimate_index] - gt_timestamps[gt_index]
        if abs(difference) <= tolerance:
            estimate_indices.append(estimate_index)
            gt_indices.append(gt_index)
            estimate_index += 1
            gt_index += 1
        elif difference < 0.0:
            estimate_index += 1
        else:
            gt_index += 1
    if not estimate_indices:
        raise ValueError("Estimate and ground truth have no matching timestamps")
    return (
        np.asarray(estimate_indices, dtype=np.int64),
        np.asarray(gt_indices, dtype=np.int64),
    )


def select_trajectory_samples(trajectory, indices: np.ndarray) -> Trajectory:
    return Trajectory([trajectory.samples[index] for index in indices])


def compute_pose_metrics(
    estimate,
    ground_truth,
    rpe_delta_seconds: float,
) -> PoseMetrics:
    """Compute absolute pose errors and fixed-time-delta relative pose errors."""
    if len(estimate) != len(ground_truth) or len(estimate) == 0:
        raise ValueError("Estimate and ground truth must have equal non-zero length")

    position_error = np.linalg.norm(
        estimate.positions - ground_truth.positions,
        axis=1,
    )
    rotation_error = np.asarray(
        [
            rotation_angle_deg(gt.pose.R.T @ est.pose.R)
            for est, gt in zip(estimate.samples, ground_truth.samples)
        ],
        dtype=np.float64,
    )
    starts, ends = fixed_delta_pairs(estimate.timestamps, rpe_delta_seconds)
    translation_rpe, rotation_rpe = relative_pose_errors(
        estimate,
        ground_truth,
        starts,
        ends,
    )

    return PoseMetrics(
        position_mean=float(np.mean(position_error)),
        position_rmse=_rmse(position_error),
        position_median=float(np.median(position_error)),
        position_max=float(np.max(position_error)),
        rotation_mean_deg=float(np.mean(rotation_error)),
        rotation_rmse_deg=_rmse(rotation_error),
        rotation_median_deg=float(np.median(rotation_error)),
        rotation_max_deg=float(np.max(rotation_error)),
        rpe_pair_count=len(starts),
        rpe_delta_median_s=float(
            np.median(estimate.timestamps[ends] - estimate.timestamps[starts])
        ),
        rpe_translation_rmse=_rmse(translation_rpe),
        rpe_rotation_rmse_deg=_rmse(rotation_rpe),
    )


def fixed_delta_pairs(timestamps: np.ndarray, delta_seconds: float) -> tuple:
    """Pair every pose with the closest later pose at a fixed time delta."""
    timestamps = np.asarray(timestamps, dtype=np.float64)
    delta_seconds = float(delta_seconds)
    if delta_seconds <= 0.0:
        raise ValueError("RPE time delta must be positive")
    if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("RPE requires at least two strictly increasing timestamps")

    targets = timestamps + delta_seconds
    right = np.searchsorted(timestamps, targets)
    starts = np.flatnonzero(right < len(timestamps))
    ends = right[starts]
    left = ends - 1
    use_left = (left > starts) & (
        np.abs(timestamps[left] - targets[starts])
        < np.abs(timestamps[ends] - targets[starts])
    )
    ends[use_left] = left[use_left]
    if len(starts) == 0:
        raise ValueError("Trajectory is shorter than the requested RPE time delta")
    return starts.astype(np.int64), ends.astype(np.int64)


def relative_pose_errors(estimate, ground_truth, starts, ends) -> tuple:
    translation = np.empty(len(starts), dtype=np.float64)
    rotation = np.empty(len(starts), dtype=np.float64)
    for output_index, (start, end) in enumerate(zip(starts, ends)):
        T_est_start = estimate.samples[start].pose.as_matrix()
        T_est_end = estimate.samples[end].pose.as_matrix()
        T_gt_start = ground_truth.samples[start].pose.as_matrix()
        T_gt_end = ground_truth.samples[end].pose.as_matrix()
        estimate_delta = invert_transform(T_est_start) @ T_est_end
        gt_delta = invert_transform(T_gt_start) @ T_gt_end
        error = invert_transform(gt_delta) @ estimate_delta
        translation[output_index] = np.linalg.norm(error[:3, 3])
        rotation[output_index] = rotation_angle_deg(error[:3, :3])
    return translation, rotation


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))
