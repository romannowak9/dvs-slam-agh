from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from event_slam.core.imu import ImuCoverageError, imu_rotation_between_camera_times
from event_slam.core.types import StereoEventWindow
from event_slam.core.velocity import compute_velocity_trajectory
from event_slam.events.imu_motion_compensation import (
    compensate_stereo_window_rotation,
)


@dataclass
class EvSlamStereoVOSummary:
    """
    Summary of one VO pipeline run.
    """

    processed_frames: int
    successful_steps: int
    failed_frames: int
    median_inliers: float
    final_position: np.ndarray
    velocity_samples: int = 0
    motion_compensated_frames: int = 0
    motion_compensation_failed: int = 0
    imu_prior_available_frames: int = 0
    imu_rejected_steps: int = 0
    keyframe_count: int = 0
    landmark_count: int = 0


class EvSlamStereoVOPipeline:
    """
    Connect configured event processing, rectification and stereo VO modules.

    Configuration parsing and module construction are intentionally kept outside
    this class. The pipeline only passes each stage's output to the next stage.
    """

    def __init__(
        self,
        window_builder,
        aggregator,
        rectifier,
        slam,
        calibration,
        background_filter=None,
        imu_timestamps=None,
        imu_angular_velocities=None,
        imu_time_offset: float = 0.0,
        motion_compensation_cfg=None,
        rotation_prior_cfg=None,
        R_output_from_pnp_camera=None,
        num_frames: int = 0,
        velocity_smoothing_window: int = 1,
        velocity_smoothing_poly_order: int = 2,
        frame_callback=None,
    ) -> None:
        self.window_builder = window_builder
        self.aggregator = aggregator
        self.rectifier = rectifier
        self.slam = slam
        self.calibration = calibration
        self.background_filter = background_filter

        self.imu_timestamps = imu_timestamps
        self.imu_angular_velocities = imu_angular_velocities
        self.imu_time_offset = float(imu_time_offset)
        self.motion_compensation_cfg = motion_compensation_cfg
        self.rotation_prior_cfg = rotation_prior_cfg
        self.R_output_from_pnp_camera = R_output_from_pnp_camera

        self.num_frames = int(num_frames)
        self.velocity_smoothing_window = int(velocity_smoothing_window)
        self.velocity_smoothing_poly_order = int(velocity_smoothing_poly_order)
        self.frame_callback = frame_callback

        self.previous_timestamp = None
        self.last_motion_compensation = None
        self.velocity_trajectory = None

        self.motion_compensated_frames = 0
        self.motion_compensation_failed = 0
        self.imu_prior_available_frames = 0
        self.imu_rejected_steps = 0

    @property
    def trajectory(self):
        """
        Return the trajectory owned by StereoPnPSLAM.
        """
        return self.slam.trajectory

    def run(self) -> EvSlamStereoVOSummary:
        for frame_index, window in enumerate(self.window_builder.iter_windows()):
            if self.num_frames > 0 and len(self.slam.results) >= self.num_frames:
                break

            self.process_window(window, frame_index)

        return self.get_summary()

    def process_window(
        self,
        window: StereoEventWindow,
        frame_index: int,
    ):
        if self.background_filter is not None:
            window = self.background_filter.filter(window)

        window = self._compensate_window(window)
        stereo_frame = self.aggregator.aggregate_stereo_window(window)
        left_rectified, right_rectified = self.rectifier.rectify_pair(
            stereo_frame.left.image,
            stereo_frame.right.image,
        )

        timestamp = stereo_frame.timestamp
        result = self.slam.process(
            left_rectified=left_rectified,
            right_rectified=right_rectified,
            timestamp=timestamp,
            imu_rotation_prior=self._make_imu_rotation_prior(timestamp),
        )

        self.previous_timestamp = timestamp
        self.velocity_trajectory = None
        if result.imu_rejected:
            self.imu_rejected_steps += 1

        if self.frame_callback is not None:
            self.frame_callback(
                frame_index,
                window,
                result,
                self.last_motion_compensation,
            )

        return result

    def compute_velocity(self):
        """
        Compute velocity from the current VO trajectory.
        """
        self.velocity_trajectory = compute_velocity_trajectory(
            trajectory=self.trajectory,
            smoothing_window_size=self.velocity_smoothing_window,
            smoothing_poly_order=self.velocity_smoothing_poly_order,
        )
        return self.velocity_trajectory

    def get_summary(self) -> EvSlamStereoVOSummary:
        """
        Return summary statistics for the current run.
        """
        slam_summary = self.slam.get_summary()

        if self.velocity_trajectory is None:
            velocity_samples = 0
        else:
            velocity_samples = len(self.velocity_trajectory)

        return EvSlamStereoVOSummary(
            processed_frames=slam_summary.processed_frames,
            successful_steps=slam_summary.successful_steps,
            failed_frames=slam_summary.failed_frames,
            median_inliers=slam_summary.median_inliers,
            final_position=slam_summary.final_position,
            velocity_samples=velocity_samples,
            motion_compensated_frames=self.motion_compensated_frames,
            motion_compensation_failed=self.motion_compensation_failed,
            imu_prior_available_frames=self.imu_prior_available_frames,
            imu_rejected_steps=self.imu_rejected_steps,
            keyframe_count=slam_summary.keyframe_count,
            landmark_count=slam_summary.landmark_count,
        )

    def _compensate_window(self, window: StereoEventWindow) -> StereoEventWindow:
        self.last_motion_compensation = None

        if self.motion_compensation_cfg is None:
            return window

        # Edge event windows can extend beyond the recorded IMU time range.
        # Keep the original window so visual odometry can continue without motion compensation.
        try:
            result = compensate_stereo_window_rotation(
                window=window,
                left_camera=self.calibration.left,
                right_camera=self.calibration.right,
                imu_timestamps=self.imu_timestamps,
                angular_velocities=self.imu_angular_velocities,
                T_C_left_imu=self.calibration.T_C_left_imu,
                T_C_right_imu=self.calibration.T_C_right_imu,
                timeshift_left_imu=self.calibration.timeshift_left_imu,
                timeshift_right_imu=self.calibration.timeshift_right_imu,
                imu_time_offset=self.imu_time_offset,
                reference_time=self.motion_compensation_cfg.get(
                    "reference_time",
                    "middle",
                ),
                num_time_bins=int(self.motion_compensation_cfg.get("time_bins", 32)),
            )
        except ImuCoverageError as exc:
            self.motion_compensation_failed += 1
            print(f"IMU motion compensation skipped: {exc}")
            return window

        self.last_motion_compensation = result
        self.motion_compensated_frames += 1

        return result.window

    def _make_imu_rotation_prior(self, timestamp: float):
        if self.rotation_prior_cfg is None or self.previous_timestamp is None:
            return None

        # A frame interval can fall outside IMU coverage at sequence boundaries.
        # The rotation prior is optional, so let PnP continue without it.
        try:
            R_imu = imu_rotation_between_camera_times(
                imu_timestamps=self.imu_timestamps,
                angular_velocities=self.imu_angular_velocities,
                camera_start_time=self.previous_timestamp,
                camera_end_time=timestamp,
                T_C_imu=self.calibration.T_C_left_imu,
                timeshift_cam_imu=self.calibration.timeshift_left_imu,
                imu_time_offset=self.imu_time_offset,
                R_output_from_camera=self.R_output_from_pnp_camera,
            )
        except ImuCoverageError:
            return None

        self.imu_prior_available_frames += 1

        return R_imu
