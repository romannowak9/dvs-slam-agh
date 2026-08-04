#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"

if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))


from event_slam.calibration.kalibr_parser import load_stereo_calibration
from event_slam.calibration.stereo_rectifier import StereoRectifier
from event_slam.datasets.evslam_reader import (
    DEFAULT_LEFT_EVENT_TOPIC,
    DEFAULT_RIGHT_EVENT_TOPIC,
    EvSlamRosbagReader,
)
from event_slam.events.event_aggregator import (
    EventFrameAggregator,
    EventFrameMode,
    PolarityMode,
)
from event_slam.events.event_filter import StereoBackgroundActivityFilter
from event_slam.events.event_window import StereoEventWindowBuilder
from event_slam.io.result_io import write_vo_csv
from event_slam.vo.feature_tracker import FeatureDetectorMode
from event_slam.vo.stereo_pnp_vo import StereoPnPVO

from event_slam.debug.visualization import (
    colorize_event_frame,
    draw_points,
    draw_text_bar,
    format_value,
    save_image,
    show_image,
)


def main() -> None:
    args = parse_args()

    if args.save_debug:
        args.debug_dir.mkdir(parents=True, exist_ok=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    calibration = load_stereo_calibration(args.camera_yaml)
    image_shape = calibration.left.image_shape

    reader = EvSlamRosbagReader(
        bag_path=args.bag,
        left_event_topic=args.left_topic,
        right_event_topic=args.right_topic,
    )

    builder = StereoEventWindowBuilder.from_reader(
        reader=reader,
        time_window=args.time_window,
        t_start=args.t_start,
        t_end=args.t_end,
        drop_empty_windows=True,
    )

    aggregator = EventFrameAggregator(
        image_shape=image_shape,
        mode=args.mode,
        polarity_mode=args.polarity_mode,
        tau=args.tau,
    )

    rectifier = StereoRectifier(
        calibration=calibration,
        image_shape=image_shape,
        alpha=args.alpha,
        interpolation="nearest",
    )

    vo = StereoPnPVO(
        K=rectifier.K_left_rectified,
        P1=rectifier.P1,
        P2=rectifier.P2,
        feature_tracker_params={
            "detector": args.detector,
            "min_features": args.min_features,
            "max_features": args.max_features,
            "fast_threshold": args.fast_threshold,
            "use_forward_backward_check": args.forward_backward_check,
            "fb_threshold": args.fb_threshold,
        },
        stereo_depth_params={
            "epipolar_threshold": args.epipolar_threshold,
            "min_disparity": args.min_disparity,
            "max_disparity": args.max_disparity,
            "min_depth": args.min_depth,
            "max_depth": args.max_depth,
        },
        min_pnp_points=args.min_pnp_points,
        min_pnp_inliers=args.min_pnp_inliers,
        min_pnp_inlier_ratio=args.min_pnp_inlier_ratio,
        max_pnp_reprojection_median=args.max_pnp_reprojection_median,
        max_translation_step=args.max_translation_step,
        max_rotation_step_deg=args.max_rotation_step_deg,
        pnp_reprojection_error=args.pnp_reprojection_error,
        pnp_confidence=args.pnp_confidence,
        pnp_iterations=args.pnp_iterations,
    )

    background_filter = None

    if args.use_baf:
        background_filter = StereoBackgroundActivityFilter(
            image_shape=image_shape,
            time_window=args.baf_time_window,
            radius=args.baf_radius,
            min_neighbors=args.baf_min_neighbors,
        )

    processed = 0

    for frame_index, window in enumerate(builder.iter_windows()):
        if args.num_frames > 0 and processed >= args.num_frames:
            break

        if args.use_baf:
            window = background_filter.filter(window)

        stereo_frame = aggregator.aggregate_stereo_window(window)

        rect_left, rect_right = rectifier.rectify_pair(
            stereo_frame.left.image,
            stereo_frame.right.image,
        )

        timestamp = 0.5 * (window.t_start + window.t_end)

        result = vo.process(
            left_rectified=rect_left,
            right_rectified=rect_right,
            timestamp=timestamp,
        )

        print_frame_result(frame_index, result)

        if args.save_debug or args.display:
            debug_image = make_debug_image(rect_left, result)

            if args.save_debug:
                output_path = args.debug_dir / f"frame_{frame_index:05d}.png"
                save_image(output_path, debug_image)

            if args.display:
                keep_running = show_image(
                    window_name="run_stereo_pnp_vo",
                    image=debug_image,
                    delay_ms=args.display_delay_ms,
                )

                if not keep_running:
                    break

        processed += 1

    if args.display:
        cv2.destroyAllWindows()

    csv_path = args.output_dir / "trajectory.csv"

    write_vo_csv(vo.results, csv_path)

    print_summary(vo.get_summary())
    print()
    print(f"trajectory_csv: {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run stereo PnP visual odometry on EvSLAM data."
    )

    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument("--camera-yaml", required=True, type=Path)

    parser.add_argument("--left-topic", default=DEFAULT_LEFT_EVENT_TOPIC)
    parser.add_argument("--right-topic", default=DEFAULT_RIGHT_EVENT_TOPIC)

    parser.add_argument("--time-window", default=0.0333333333, type=float)
    parser.add_argument("--num-frames", default=500, type=int, help="Number of frames to process. Use 0 to process the whole bag.",)

    parser.add_argument(
        "--mode",
        default=EventFrameMode.STANDARD.value,
        choices=[mode.value for mode in EventFrameMode],
    )

    parser.add_argument(
        "--polarity-mode",
        default=PolarityMode.BOTH.value,
        choices=[mode.value for mode in PolarityMode],
    )

    parser.add_argument("--tau", default=0.03, type=float)

    parser.add_argument("--use-baf", action="store_true")
    parser.add_argument("--baf-time-window", default=1.0 / 24.0, type=float)
    parser.add_argument("--baf-radius", default=2, type=int)
    parser.add_argument("--baf-min-neighbors", default=1, type=int)

    parser.add_argument("--t-start", default=None, type=float)
    parser.add_argument("--t-end", default=None, type=float)

    parser.add_argument("--alpha", default=0.0, type=float)

    parser.add_argument(
        "--detector",
        default=FeatureDetectorMode.FAST.value,
        choices=[mode.value for mode in FeatureDetectorMode],
    )

    parser.add_argument("--min-features", default=250, type=int)
    parser.add_argument("--max-features", default=1000, type=int)
    parser.add_argument("--fast-threshold", default=25, type=int)

    parser.add_argument("--forward-backward-check", action="store_true")
    parser.add_argument("--fb-threshold", default=1.0, type=float)

    parser.add_argument("--epipolar-threshold", default=2.0, type=float)
    parser.add_argument("--min-disparity", default=0.5, type=float)
    parser.add_argument("--max-disparity", default=250.0, type=float)

    parser.add_argument("--min-depth", default=0.05, type=float)
    parser.add_argument("--max-depth", default=100.0, type=float)

    parser.add_argument("--min-pnp-points", default=20, type=int)
    parser.add_argument("--min-pnp-inliers", default=30, type=int)
    parser.add_argument("--min-pnp-inlier-ratio", default=0.15, type=float)

    parser.add_argument("--max-pnp-reprojection-median", default=3.0, type=float)
    parser.add_argument("--max-translation-step", default=0.5, type=float)
    parser.add_argument("--max-rotation-step-deg", default=15.0, type=float)

    parser.add_argument("--pnp-reprojection-error", default=3.0, type=float)
    parser.add_argument("--pnp-confidence", default=0.999, type=float)
    parser.add_argument("--pnp-iterations", default=100, type=int)

    parser.add_argument("--display", action="store_true")
    parser.add_argument("--display-delay-ms", default=1, type=int)

    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument(
        "--debug-dir",
        default=Path("outputs/stereo_pnp_vo/debug"),
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        default=Path("outputs/stereo_pnp_vo"),
        type=Path,
    )

    return parser.parse_args()


def print_frame_result(frame_index: int, result) -> None:
    t = result.T_W_Cleft[:3, 3]

    print(
        f"frame {frame_index:05d}: "
        f"success={result.success}, "
        f"tracks={result.track_count}, "
        f"pnp={result.pnp_point_count}, "
        f"inliers={result.pnp_inlier_count}, "
        f"err_med={format_value(result.reprojection_error_median)}, "
        f"depth={result.depth_count}, "
        f"pos=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}], "
        f"msg={result.message}"
    )


def print_summary(summary) -> None:
    print()
    print("VO summary")
    print("=" * 80)
    print(f"processed_frames: {summary.processed_frames}")
    print(f"successful_steps: {summary.successful_steps}")
    print(f"failed_frames: {summary.failed_frames}")
    print(f"median_inliers: {format_value(summary.median_inliers)}")
    print(
        "final_position: "
        f"[{summary.final_position[0]:.6f}, "
        f"{summary.final_position[1]:.6f}, "
        f"{summary.final_position[2]:.6f}]"
    )


def make_debug_image(left_gray: np.ndarray, result) -> np.ndarray:
    image = colorize_event_frame(left_gray)
    draw_points(image, result.tracked_points_curr, color=(80, 80, 80))
    draw_points(image, result.pnp_points_curr, color=(0, 255, 255))
    draw_points(
        image,
        result.pnp_inlier_points_curr,
        color=(0, 255, 0),
        radius=3,
        thickness=1,
    )
    draw_status_bar(image, result)
    return image


def draw_status_bar(image: np.ndarray, result) -> None:
    text = (
        f"success={int(result.success)} "
        f"tracks={result.track_count} "
        f"pnp={result.pnp_point_count} "
        f"inliers={result.pnp_inlier_count} "
        f"err={format_value(result.reprojection_error_median)}"
    )
    draw_text_bar(image, text)


if __name__ == "__main__":
    main()
