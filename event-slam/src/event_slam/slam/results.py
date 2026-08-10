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

    tracking_state: str = "TRACKING"
    reference_keyframe_id: int = -1
    T_C_ref_C_frame: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float64)
    )
    is_keyframe: bool = False
    loop_candidate_count: int = 0
    loop_candidate_id: int = -1
    loop_match_count: int = 0
    loop_accepted: bool = False
    loop_relative_scale: float = np.nan
    relocalized: bool = False
    graph_cost_before: float = np.nan
    graph_cost_after: float = np.nan


@dataclass
class StereoPnPSLAMSummary:
    processed_frames: int
    successful_steps: int
    failed_frames: int
    median_inliers: float
    final_position: np.ndarray
    keyframe_count: int = 0
    landmark_count: int = 0
    accepted_loop_count: int = 0
    relocalization_count: int = 0
    graph_cost_before: float = np.nan
    graph_cost_after: float = np.nan
