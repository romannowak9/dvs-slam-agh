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
from event_slam.core.types import StereoEventWindow
from event_slam.datasets.evslam_reader import (
    DEFAULT_LEFT_EVENT_TOPIC,
    DEFAULT_RIGHT_EVENT_TOPIC,
    EvSlamRosbagReader,
)
from event_slam.events.event_aggregator import (
    BACKGROUND_INTENSITY,
    EventFrameAggregator,
    EventFrameMode,
    PolarityMode,
)
from event_slam.events.event_filter import BackgroundActivityFilter
from event_slam.events.event_window import StereoEventWindowBuilder
from event_slam.vo.feature_tracker import FeatureDetectorMode, FeatureTracker
from event_slam.vo.stereo_depth import StereoDepthEstimator

from event_slam.debug.visualization import (
    colorize_event_frame,
    format_value,
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

    left_baf = None
    right_baf = None

    if args.use_baf:
        left_baf = BackgroundActivityFilter(
            image_shape=image_shape,
            time_window=args.baf_time_window,
            radius=args.baf_radius,
            min_neighbors=args.baf_min_neighbors,
        )

        right_baf = BackgroundActivityFilter(
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
            window = StereoEventWindow(
                t_start=window.t_start,
                t_end=window.t_end,
                left=left_baf.filter(window.left),
                right=right_baf.filter(window.right),
            )

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

    debug = np.concatenate((left_color, right_color), axis=1)
    width = left_color.shape[1]

    count = min(len(depth_result.points_2d_left), int(max_draw_matches))

    if count == 0:
        return debug

    depths = depth_result.points_3d_left_camera[:count, 2]
    depth_min = float(np.min(depths))
    depth_max = float(np.max(depths))
    denom = max(1e-9, depth_max - depth_min)

    for index in range(count):
        left_pt = depth_result.points_2d_left[index]
        right_pt = depth_result.points_2d_right[index]
        depth = depth_result.points_3d_left_camera[index, 2]

        intensity = int(round(255.0 * (depth - depth_min) / denom))
        color = (0, 255 - intensity, 255)

        x0, y0 = int(round(left_pt[0])), int(round(left_pt[1]))
        x1, y1 = int(round(right_pt[0])) + width, int(round(right_pt[1]))

        cv2.circle(debug, (x0, y0), 2, color, -1, cv2.LINE_AA)
        cv2.circle(debug, (x1, y1), 2, color, -1, cv2.LINE_AA)
        cv2.line(debug, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)

    return debug


def draw_stats(image: np.ndarray, depth_result) -> None:
    stats = depth_result.stats

    text = (
        f"in={stats.input_count} "
        f"matched={stats.matched_count} "
        f"3d={stats.triangulated_count} "
        f"z_med={format_value(stats.depth_median)}"
    )

    cv2.rectangle(image, (0, 0), (image.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(
        image,
        text,
        (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


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