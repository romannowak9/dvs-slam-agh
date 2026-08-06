from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from event_slam.core.geometry import empty_points


@dataclass
class StereoPnPSLAMResult:
    """Result of processing one rectified stereo frame pair."""

    timestamp: float
    success: bool
    initialized: bool
    T_W_Cleft: np.ndarray

    track_count: int = 0
    pnp_point_count: int = 0
    pnp_inlier_count: int = 0
    pose_source: str = "none"
    map_point_count: int = 0
    map_inlier_count: int = 0
    local_landmark_count: int = 0
    map_descriptor_match_count: int = 0
    map_message: str = ""
    new_feature_count: int = 0

    reprojection_error_mean: float = np.nan
    reprojection_error_median: float = np.nan
    pnp_rotation_step_deg: float = np.nan
    imu_rotation_step_deg: float = np.nan
    pnp_imu_rotation_error_deg: float = np.nan
    imu_rotation_consistent: bool = False
    imu_rejected: bool = False

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
class StereoPnPSLAMSummary:
    processed_frames: int
    successful_steps: int
    failed_frames: int
    median_inliers: float
    final_position: np.ndarray
    keyframe_count: int = 0
    landmark_count: int = 0
