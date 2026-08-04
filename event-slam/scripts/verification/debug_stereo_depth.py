#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import cv2


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
from event_slam.vo.feature_tracker import FeatureDetectorMode, FeatureTracker
from event_slam.vo.stereo_depth import StereoDepthEstimator

from event_slam.debug.visualization import (
    colorize_event_frame,
    draw_stereo_matches,
    draw_text_bar,
    format_value,
    make_side_by_side,
    save_image,
    show_image,
)


def main() -> None:
    args = parse_args()

    if not args.display:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None

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

    tracker = FeatureTracker(
        detector=args.detector,
        min_features=args.min_features,
        max_features=args.max_features,
        fast_threshold=args.fast_threshold,
    )

    depth_estimator = StereoDepthEstimator(
        P1=rectifier.P1,
        P2=rectifier.P2,
        epipolar_threshold=args.epipolar_threshold,
        min_disparity=args.min_disparity,
        max_disparity=args.max_disparity,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
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

        rect_left_gray, rect_right_gray = rectifier.rectify_pair(
            stereo_frame.left.image,
            stereo_frame.right.image,
        )

        tracking_result = tracker.process(rect_left_gray)
        left_points = tracking_result.active_points

        depth_result = depth_estimator.estimate(
            left_img=rect_left_gray,
            right_img=rect_right_gray,
            left_points=left_points,
        )

        debug_image = make_debug_image(
            left_gray=rect_left_gray,
            right_gray=rect_right_gray,
            depth_result=depth_result,
            max_draw_matches=args.max_draw_matches,
        )

        draw_stats(debug_image, depth_result)

        if args.display:
            keep_running = show_image(
                window_name="debug_stereo_depth",
                image=debug_image,
                delay_ms=args.display_delay_ms,
            )

            if not keep_running:
                break
        else:
            output_path = output_dir / f"frame_{frame_index:05d}.png"
            save_image(output_path, debug_image)

        print_frame_stats(frame_index, window, tracking_result, depth_result)

        processed += 1

    if args.display:
        cv2.destroyAllWindows()

    print()
    print(f"processed_frames: {processed}")

    if not args.display:
        print(f"output_dir: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug sparse stereo matching and triangulation on event frames."
    )

    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument("--camera-yaml", required=True, type=Path)

    parser.add_argument("--left-topic", default=DEFAULT_LEFT_EVENT_TOPIC)
    parser.add_argument("--right-topic", default=DEFAULT_RIGHT_EVENT_TOPIC)

    parser.add_argument("--time-window", default=0.0333333333, type=float)
    parser.add_argument("--num-frames", default=100, type=int, help="Number of frames to process. Use 0 to process the whole bag.",)

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

    parser.add_argument("--epipolar-threshold", default=2.0, type=float)
    parser.add_argument("--min-disparity", default=0.5, type=float)
    parser.add_argument("--max-disparity", default=250.0, type=float)

    parser.add_argument("--min-depth", default=0.05, type=float)
    parser.add_argument("--max-depth", default=100.0, type=float)

    parser.add_argument("--max-draw-matches", default=200, type=int)

    parser.add_argument("--display", action="store_true")
    parser.add_argument("--display-delay-ms", default=1, type=int)

    parser.add_argument(
        "--output-dir",
        default="outputs/debug_stereo_depth",
        type=Path,
    )

    return parser.parse_args()


def make_debug_image(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    depth_result,
    max_draw_matches: int,
) -> np.ndarray:
    left_color = colorize_event_frame(left_gray)
    right_color = colorize_event_frame(right_gray)
    debug = make_side_by_side(left_color, right_color)

    draw_stereo_matches(
        image=debug,
        points_left=depth_result.points_2d_left,
        points_right=depth_result.points_2d_right,
        left_width=left_color.shape[1],
        points_3d=depth_result.points_3d_left_camera,
        max_matches=max_draw_matches,
    )

    return debug


def draw_stats(image: np.ndarray, depth_result) -> None:
    stats = depth_result.stats
    text = (
        f"in={stats.input_count} "
        f"matched={stats.matched_count} "
        f"3d={stats.triangulated_count} "
        f"z_med={format_value(stats.depth_median)}"
    )
    draw_text_bar(image, text)


def print_frame_stats(frame_index: int, window, tracking_result, depth_result) -> None:
    stats = depth_result.stats

    print(
        f"frame {frame_index:05d}: "
        f"tracks_active={tracking_result.active_count}, "
        f"stereo_in={stats.input_count}, "
        f"matched={stats.matched_count}, "
        f"triangulated={stats.triangulated_count}, "
        f"disp_med={format_value(stats.disparity_median)}, "
        f"z_med={format_value(stats.depth_median)}, "
        f"t=[{window.t_start:.9f}, {window.t_end:.9f})"
    )


if __name__ == "__main__":
    main()
