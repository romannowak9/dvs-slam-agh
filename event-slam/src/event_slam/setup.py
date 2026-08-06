from __future__ import annotations

from event_slam.calibration.kalibr_parser import (
    load_imu_calibration,
    load_stereo_calibration,
)
from event_slam.calibration.stereo_rectifier import StereoRectifier
from event_slam.core.geometry import as_float_array
from event_slam.datasets.evslam_reader import EvSlamRosbagReader
from event_slam.debug.visualization import print_vo_frame
from event_slam.events.event_aggregator import EventFrameAggregator
from event_slam.events.event_filter import StereoBackgroundActivityFilter
from event_slam.events.event_window import StereoEventWindowBuilder
from event_slam.pipeline import EvSlamStereoVOPipeline
from event_slam.slam.stereo_pnp import StereoPnPSLAM


def create_pipeline(config: dict) -> EvSlamStereoVOPipeline:
    """Construct the configured modules and connect them into the VO pipeline."""
    dataset_cfg = config.get("dataset", {})
    processing_cfg = config.get("processing", {})
    aggregation_cfg = config.get("aggregation", {})
    baf_cfg = config.get("baf", {})
    imu_cfg = config.get("imu", {})
    motion_compensation_cfg = imu_cfg.get("motion_compensation", {})
    rotation_prior_cfg = imu_cfg.get("rotation_prior", {})
    rectification_cfg = config.get("rectification", {})
    feature_tracker_cfg = config.get("feature_tracker", {})
    stereo_depth_cfg = config.get("stereo_depth", {})
    pnp_cfg = config.get("pnp", {})
    slam_cfg = config.get("slam", {})
    velocity_cfg = config.get("velocity", {})

    calibration = load_stereo_calibration(dataset_cfg["camera_yaml"])
    image_shape = calibration.left.image_shape
    reader = EvSlamRosbagReader(
        bag_path=dataset_cfg["bag_path"],
        left_event_topic=dataset_cfg["left_event_topic"],
        right_event_topic=dataset_cfg["right_event_topic"],
    )
    window_builder = StereoEventWindowBuilder.from_reader(
        reader=reader,
        time_window=float(processing_cfg.get("time_window", 0.007)),
        t_start=processing_cfg.get("t_start"),
        t_end=processing_cfg.get("t_end"),
        drop_empty_windows=True,
    )
    aggregator = EventFrameAggregator(
        image_shape=image_shape,
        mode=aggregation_cfg.get("mode", "exponential"),
        polarity_mode=aggregation_cfg.get("polarity_mode", "both"),
        tau=float(aggregation_cfg.get("tau", 0.006)),
    )
    rectifier = StereoRectifier(
        calibration=calibration,
        image_shape=image_shape,
        alpha=float(rectification_cfg.get("alpha", 0.0)),
        interpolation=rectification_cfg.get("interpolation", "nearest"),
    )

    background_filter = None
    if bool(baf_cfg.get("enabled", False)):
        background_filter = StereoBackgroundActivityFilter(
            image_shape=image_shape,
            time_window=float(baf_cfg.get("time_window", 0.005)),
            radius=int(baf_cfg.get("radius", 2)),
            min_neighbors=int(baf_cfg.get("min_neighbors", 5)),
        )

    R_output_from_pnp_camera = pnp_cfg.get("R_output_from_pnp_camera")
    if R_output_from_pnp_camera is not None:
        R_output_from_pnp_camera = as_float_array(
            R_output_from_pnp_camera,
            (3, 3),
            "R_output_from_pnp_camera",
        )

    slam = StereoPnPSLAM(
        K=rectifier.K_left_rectified,
        P1=rectifier.P1,
        P2=rectifier.P2,
        R_rect_left_from_left=rectifier.R1,
        feature_tracker_params={
            "detector": feature_tracker_cfg.get("detector", "fast"),
            "max_features": int(feature_tracker_cfg.get("max_features", 1000)),
            "fast_threshold": int(feature_tracker_cfg.get("fast_threshold", 25)),
            "use_forward_backward_check": bool(
                feature_tracker_cfg.get("forward_backward_check", False)
            ),
            "fb_threshold": float(feature_tracker_cfg.get("fb_threshold", 1.0)),
        },
        stereo_depth_params={
            "epipolar_threshold": float(
                stereo_depth_cfg.get("epipolar_threshold", 2.0)
            ),
            "min_disparity": float(stereo_depth_cfg.get("min_disparity", 0.5)),
            "max_disparity": stereo_depth_cfg.get("max_disparity", 250.0),
            "min_depth": float(stereo_depth_cfg.get("min_depth", 0.05)),
            "max_depth": stereo_depth_cfg.get("max_depth", 100.0),
        },
        min_pnp_points=int(pnp_cfg.get("min_pnp_points", 20)),
        min_pnp_inliers=int(pnp_cfg.get("min_pnp_inliers", 30)),
        min_pnp_inlier_ratio=float(pnp_cfg.get("min_pnp_inlier_ratio", 0.15)),
        max_pnp_reprojection_median=float(
            pnp_cfg.get("max_pnp_reprojection_median", 3.0)
        ),
        max_translation_step=float(pnp_cfg.get("max_translation_step", 0.5)),
        max_rotation_step_deg=float(pnp_cfg.get("max_rotation_step_deg", 15.0)),
        pnp_reprojection_error=float(pnp_cfg.get("pnp_reprojection_error", 3.0)),
        pnp_confidence=float(pnp_cfg.get("pnp_confidence", 0.999)),
        pnp_iterations=int(pnp_cfg.get("pnp_iterations", 100)),
        R_output_from_pnp_camera=R_output_from_pnp_camera,
        imu_rotation_prior_max_error_deg=float(
            rotation_prior_cfg.get("max_error_deg", 3.0)
        ),
        imu_rotation_prior_reject_bad_pnp=bool(
            rotation_prior_cfg.get("reject_bad_pnp", False)
        ),
        slam_params=slam_cfg,
    )

    imu_enabled = bool(imu_cfg.get("enabled", False))
    motion_compensation_enabled = imu_enabled and bool(
        motion_compensation_cfg.get("enabled", False)
    )
    rotation_prior_enabled = imu_enabled and bool(
        rotation_prior_cfg.get("enabled", False)
    )
    imu_timestamps = None
    imu_angular_velocities = None
    imu_time_offset = 0.0

    if motion_compensation_enabled or rotation_prior_enabled:
        imu_yaml = (
            imu_cfg.get("calibration_yaml")
            or dataset_cfg.get("imu_yaml")
            or dataset_cfg.get("imu_calibration")
        )
        if imu_yaml is None:
            raise ValueError(
                "IMU is enabled, but IMU calibration YAML was not provided. "
                "Use imu.calibration_yaml or dataset.imu_yaml."
            )

        imu_calibration = load_imu_calibration(imu_yaml)
        imu_topic = imu_cfg.get("topic") or imu_calibration.topic
        if imu_topic is None:
            raise ValueError(
                "IMU is enabled, but IMU topic was not provided and could not "
                "be read from IMU calibration."
            )

        imu_timestamps, imu_angular_velocities = reader.load_imu_gyro(topic=imu_topic)
        imu_time_offset = imu_calibration.time_offset or 0.0

    return EvSlamStereoVOPipeline(
        window_builder=window_builder,
        aggregator=aggregator,
        rectifier=rectifier,
        slam=slam,
        calibration=calibration,
        background_filter=background_filter,
        imu_timestamps=imu_timestamps,
        imu_angular_velocities=imu_angular_velocities,
        imu_time_offset=imu_time_offset,
        motion_compensation_cfg=(
            motion_compensation_cfg if motion_compensation_enabled else None
        ),
        rotation_prior_cfg=rotation_prior_cfg if rotation_prior_enabled else None,
        R_output_from_pnp_camera=R_output_from_pnp_camera,
        num_frames=int(processing_cfg.get("num_frames", 0)),
        velocity_smoothing_window=int(velocity_cfg.get("smoothing_window", 1)),
        velocity_smoothing_poly_order=int(
            velocity_cfg.get("smoothing_poly_order", 2)
        ),
        frame_callback=print_vo_frame,
    )
