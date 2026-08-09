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
    EventFrameAggregator,
    EventFrameMode,
    PolarityMode,
)
from event_slam.events.event_filter import BackgroundActivityFilter
from event_slam.events.event_window import StereoEventWindowBuilder
from event_slam.vo.feature_tracker import FeatureDetectorMode, FeatureTracker

from event_slam.debug.visualization import (
    colorize_event_frame,
    draw_text_bar,
    draw_tracks,
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
        max_features=args.max_features,
        fast_threshold=args.fast_threshold,
        use_forward_backward_check=args.forward_backward_check,
        fb_threshold=args.fb_threshold,
    )

    left_baf = None

    if args.use_baf:
        left_baf = BackgroundActivityFilter(
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
                right=window.right,
            )

        left_frame = aggregator.aggregate_batch(
            batch=window.left,
            t_start=window.t_start,
            t_end=window.t_end,
        )

        rectified_left = rectifier.rectify_left(left_frame.image)
        result = tracker.process(rectified_left)

        debug_image = colorize_event_frame(rectified_left)
        draw_tracks(debug_image, result.prev_points, result.curr_points)
        draw_stats(debug_image, result)

        if args.display:
            keep_running = show_image(
                window_name="debug_feature_tracking",
                image=debug_image,
                delay_ms=args.display_delay_ms,
            )

            if not keep_running:
                break
        else:
            output_path = output_dir / f"frame_{frame_index:05d}.png"
            save_image(output_path, debug_image)

        print(
            f"frame {frame_index:05d}: "
            f"tracks={result.track_count}, "
            f"active={result.active_count}, "
            f"added={result.detected_count}, "
            f"t=[{window.t_start:.9f}, {window.t_end:.9f})"
        )

        processed += 1

    if args.display:
        cv2.destroyAllWindows()

    print()
    print(f"processed_frames: {processed}")

    if not args.display:
        print(f"output_dir: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug FAST/GFTT + LK feature tracking on rectified event frames."
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

    parser.add_argument("--max-features", default=1000, type=int)
    parser.add_argument("--fast-threshold", default=25, type=int)

    parser.add_argument("--forward-backward-check", action="store_true")
    parser.add_argument("--fb-threshold", default=1.0, type=float)

    parser.add_argument("--display", action="store_true")
    parser.add_argument("--display-delay-ms", default=1, type=int)

    parser.add_argument(
        "--output-dir",
        default="outputs/debug_feature_tracking",
        type=Path,
    )

    return parser.parse_args()


def draw_stats(image: np.ndarray, result) -> None:
    text = (
        f"tracks={result.track_count} "
        f"active={result.active_count} "
        f"added={result.detected_count}"
    )
    draw_text_bar(image, text)


if __name__ == "__main__":
    main()
