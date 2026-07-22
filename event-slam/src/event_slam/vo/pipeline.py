from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from event_slam.calibration.kalibr_parser import load_stereo_calibration
from event_slam.calibration.stereo_rectifier import StereoRectifier
from event_slam.core.types import StereoEventWindow
from event_slam.core.velocity import compute_velocity_trajectory
from event_slam.datasets.evslam_reader import EvSlamRosbagReader
from event_slam.events.event_aggregator import EventFrameAggregator
from event_slam.events.event_filter import BackgroundActivityFilter
from event_slam.events.event_window import StereoEventWindowBuilder
from event_slam.vo.stereo_pnp_vo import StereoPnPVO


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


class EvSlamStereoVOPipeline:
    """
    Offline EvSLAM stereo visual odometry pipeline.

    The pipeline connects existing project modules:

        rosbag reader
        -> stereo event windows
        -> optional background activity filtering
        -> event-frame aggregation
        -> stereo rectification
        -> StereoPnPVO
        -> optional velocity post-processing

    This class does not implement feature tracking, stereo depth, rectification,
    PnP or velocity estimation itself. It only orchestrates existing modules.
    """

    def __init__(self, config: dict) -> None:
        self.config = config

        self.dataset_cfg = self.config.get("dataset", {})
        self.processing_cfg = self.config.get("processing", {})
        self.aggregation_cfg = self.config.get("aggregation", {})
        self.baf_cfg = self.config.get("baf", {})
        self.rectification_cfg = self.config.get("rectification", {})
        self.feature_tracker_cfg = self.config.get("feature_tracker", {})
        self.stereo_depth_cfg = self.config.get("stereo_depth", {})
        self.pnp_cfg = self.config.get("pnp", {})
        self.velocity_cfg = self.config.get("velocity", {})
        self.output_cfg = self.config.get("output", {})

        self.processed_frames = 0
        self.successful_steps = 0
        self.failed_frames = 0
        self.results = []
        self.velocity_trajectory = None

        self._setup_modules()

    @property
    def trajectory(self):
        """
        Return the trajectory owned by StereoPnPVO.
        """
        return self.vo.trajectory

    def run(self) -> EvSlamStereoVOSummary:
        """
        Run the offline VO pipeline.
        """
        num_frames = int(self.processing_cfg.get("num_frames", 0))

        for frame_index, window in enumerate(self.window_builder.iter_windows()):
            if num_frames > 0 and self.processed_frames >= num_frames:
                break

            self.process_window(window, frame_index=frame_index)

        return self.get_summary()

    def process_window(
        self,
        window: StereoEventWindow,
        frame_index: int,
    ):
        """
        Process one stereo event window.
        """
        window = self._filter_window(window)

        stereo_frame = self.aggregator.aggregate_stereo_window(window)

        left_rectified, right_rectified = self.rectifier.rectify_pair(
            stereo_frame.left.image,
            stereo_frame.right.image,
        )

        timestamp = 0.5 * (window.t_start + window.t_end)

        result = self.vo.process(
            left_rectified=left_rectified,
            right_rectified=right_rectified,
            timestamp=timestamp,
        )

        self.results.append(result)
        self.velocity_trajectory = None
        self.processed_frames += 1

        if result.success:
            self.successful_steps += 1
        else:
            self.failed_frames += 1

        self._log_frame(
            frame_index=frame_index,
            timestamp=timestamp,
            window=window,
            result=result,
        )

        return result

    def compute_velocity(self):
        """
        Compute velocity from the current VO trajectory.
        """
        self.velocity_trajectory = compute_velocity_trajectory(self.trajectory)
        return self.velocity_trajectory

    def save_trajectory_output(self) -> tuple:
        """
        Save CSV trajectory file using StereoPnPVO methods.
        """
        output_dir = self.get_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = self.get_output_path("trajectory_csv", "trajectory.csv")

        self.vo.save_csv(csv_path)

        return csv_path

    def save_velocity_output(self):
        """
        Save velocity CSV if velocity output is enabled.
        """
        if not bool(self.velocity_cfg.get("enabled", False)):
            return None

        output_dir = self.get_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.velocity_trajectory is None:
            self.compute_velocity()

        output_csv = self.velocity_cfg.get("output_csv", "velocity.csv")
        output_path = _resolve_output_path(output_dir, output_csv)

        self.velocity_trajectory.save_csv(
            output_path,
            camera_frame=True,
        )

        return output_path

    def get_output_dir(self) -> Path:
        """
        Return the configured output directory.
        """
        return Path(self.output_cfg.get("output_dir", "outputs/evslam_vo"))

    def get_output_path(self, key: str, default_name: str) -> Path:
        """
        Return an output path from the output section.

        Relative paths are resolved against output_dir.
        """
        return _resolve_output_path(
            output_dir=self.get_output_dir(),
            path_value=self.output_cfg.get(key, default_name),
        )

    def get_summary(self) -> EvSlamStereoVOSummary:
        """
        Return summary statistics for the current run.
        """
        inliers = [
            result.pnp_inlier_count
            for result in self.results
            if result.pnp_inlier_count > 0
        ]

        if len(inliers) > 0:
            median_inliers = float(np.median(inliers))
        else:
            median_inliers = np.nan

        if len(self.results) > 0:
            final_position = self.results[-1].T_W_Cleft[:3, 3].copy()
        else:
            final_position = np.zeros(3, dtype=np.float64)

        if self.velocity_trajectory is None:
            velocity_samples = 0
        else:
            velocity_samples = len(self.velocity_trajectory)

        return EvSlamStereoVOSummary(
            processed_frames=self.processed_frames,
            successful_steps=self.successful_steps,
            failed_frames=self.failed_frames,
            median_inliers=median_inliers,
            final_position=final_position,
            velocity_samples=velocity_samples,
        )

    def print_summary(self) -> None:
        """
        Print summary statistics.
        """
        summary = self.get_summary()

        print()
        print("VO summary")
        print("=" * 80)
        print(f"processed_frames: {summary.processed_frames}")
        print(f"successful_steps: {summary.successful_steps}")
        print(f"failed_frames: {summary.failed_frames}")
        print(f"median_inliers: {_format_value(summary.median_inliers)}")
        print(f"velocity_samples: {summary.velocity_samples}")
        print(
            "final_position: "
            f"[{summary.final_position[0]:.6f}, "
            f"{summary.final_position[1]:.6f}, "
            f"{summary.final_position[2]:.6f}]"
        )

    def _setup_modules(self) -> None:
        calibration = load_stereo_calibration(self.dataset_cfg["camera_yaml"])
        image_shape = calibration.left.image_shape

        self.reader = EvSlamRosbagReader(
            bag_path=self.dataset_cfg["bag_path"],
            left_event_topic=self.dataset_cfg["left_event_topic"],
            right_event_topic=self.dataset_cfg["right_event_topic"],
        )

        self.window_builder = StereoEventWindowBuilder.from_reader(
            reader=self.reader,
            time_window=float(self.processing_cfg.get("time_window", 0.007)),
            t_start=self.processing_cfg.get("t_start"),
            t_end=self.processing_cfg.get("t_end"),
            drop_empty_windows=True,
        )

        self.aggregator = EventFrameAggregator(
            image_shape=image_shape,
            mode=self.aggregation_cfg.get("mode", "exponential"),
            polarity_mode=self.aggregation_cfg.get("polarity_mode", "both"),
            tau=float(self.aggregation_cfg.get("tau", 0.006)),
        )

        self.rectifier = StereoRectifier(
            calibration=calibration,
            image_shape=image_shape,
            alpha=float(self.rectification_cfg.get("alpha", 0.0)),
            interpolation=self.rectification_cfg.get("interpolation", "nearest"),
        )

        self.left_baf = None
        self.right_baf = None

        if bool(self.baf_cfg.get("enabled", False)):
            self.left_baf = BackgroundActivityFilter(
                image_shape=image_shape,
                time_window=float(self.baf_cfg.get("time_window", 0.005)),
                radius=int(self.baf_cfg.get("radius", 2)),
                min_neighbors=int(self.baf_cfg.get("min_neighbors", 5)),
            )

            self.right_baf = BackgroundActivityFilter(
                image_shape=image_shape,
                time_window=float(self.baf_cfg.get("time_window", 0.005)),
                radius=int(self.baf_cfg.get("radius", 2)),
                min_neighbors=int(self.baf_cfg.get("min_neighbors", 5)),
            )

        self.vo = StereoPnPVO(
            K=self.rectifier.K_left_rectified,
            P1=self.rectifier.P1,
            P2=self.rectifier.P2,
            feature_tracker_params=self._make_feature_tracker_params(),
            stereo_depth_params=self._make_stereo_depth_params(),
            **self._make_pnp_params(),
        )

    def _make_feature_tracker_params(self) -> dict:
        return {
            "detector": self.feature_tracker_cfg.get("detector", "fast"),
            "min_features": int(self.feature_tracker_cfg.get("min_features", 250)),
            "max_features": int(self.feature_tracker_cfg.get("max_features", 1000)),
            "fast_threshold": int(self.feature_tracker_cfg.get("fast_threshold", 25)),
            "use_forward_backward_check": bool(
                self.feature_tracker_cfg.get("forward_backward_check", False)
            ),
            "fb_threshold": float(self.feature_tracker_cfg.get("fb_threshold", 1.0)),
        }

    def _make_stereo_depth_params(self) -> dict:
        return {
            "epipolar_threshold": float(
                self.stereo_depth_cfg.get("epipolar_threshold", 2.0)
            ),
            "min_disparity": float(self.stereo_depth_cfg.get("min_disparity", 0.5)),
            "max_disparity": self.stereo_depth_cfg.get("max_disparity", 250.0),
            "min_depth": float(self.stereo_depth_cfg.get("min_depth", 0.05)),
            "max_depth": self.stereo_depth_cfg.get("max_depth", 100.0),
        }

    def _make_pnp_params(self) -> dict:
        return {
            "min_pnp_points": int(self.pnp_cfg.get("min_pnp_points", 20)),
            "min_pnp_inliers": int(self.pnp_cfg.get("min_pnp_inliers", 30)),
            "min_pnp_inlier_ratio": float(
                self.pnp_cfg.get("min_pnp_inlier_ratio", 0.15)
            ),
            "max_pnp_reprojection_median": float(
                self.pnp_cfg.get("max_pnp_reprojection_median", 3.0)
            ),
            "max_translation_step": float(
                self.pnp_cfg.get("max_translation_step", 0.5)
            ),
            "max_rotation_step_deg": float(
                self.pnp_cfg.get("max_rotation_step_deg", 15.0)
            ),
            "pnp_reprojection_error": float(
                self.pnp_cfg.get("pnp_reprojection_error", 3.0)
            ),
            "pnp_confidence": float(self.pnp_cfg.get("pnp_confidence", 0.999)),
            "pnp_iterations": int(self.pnp_cfg.get("pnp_iterations", 100)),
        }

    def _filter_window(self, window: StereoEventWindow) -> StereoEventWindow:
        if self.left_baf is None or self.right_baf is None:
            return window

        return StereoEventWindow(
            t_start=window.t_start,
            t_end=window.t_end,
            left=self.left_baf.filter(window.left),
            right=self.right_baf.filter(window.right),
        )

    def _log_frame(
        self,
        frame_index: int,
        timestamp: float,
        window: StereoEventWindow,
        result,
    ) -> None:
        t = result.T_W_Cleft[:3, 3]

        print(
            f"frame {frame_index:05d}: "
            f"timestamp={timestamp:.9f}, "
            f"events_left={len(window.left)}, "
            f"events_right={len(window.right)}, "
            f"success={result.success}, "
            f"tracks={result.track_count}, "
            f"pnp={result.pnp_point_count}, "
            f"inliers={result.pnp_inlier_count}, "
            f"err_med={_format_value(result.reprojection_error_median)}, "
            f"depth={result.depth_count}, "
            f"pos=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}], "
            f"msg={result.message}"
        )


def _resolve_output_path(output_dir: Path, path_value) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return output_dir / path


def _format_value(value) -> str:
    if value is None:
        return "None"

    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)

    if not np.isfinite(value):
        return "nan"

    return f"{value:.3f}"