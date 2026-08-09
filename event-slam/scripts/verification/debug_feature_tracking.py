#!/usr/bin/env python3

from __future__ import annotations

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
from event_slam.datasets.evslam_reader import EvSlamRosbagReader
from event_slam.events.event_aggregator import EventFrameAggregator
from event_slam.events.event_filter import BackgroundActivityFilter
from event_slam.events.event_window import StereoEventWindowBuilder
from event_slam.vo.feature_tracker import FeatureTracker

from event_slam.debug.visualization import (
    colorize_event_frame,
    draw_text_bar,
    draw_tracks,
    save_image,
    show_image,
)
from verification_config import load_args, verification_parser


def main() -> None:
    _, args = parse_args()

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


def draw_stats(image: np.ndarray, result) -> None:
    text = (
        f"tracks={result.track_count} "
        f"active={result.active_count} "
        f"added={result.detected_count}"
    )
    draw_text_bar(image, text)


def parse_args() -> tuple:
    parser = verification_parser(
        "Debug FAST/GFTT + LK tracking on rectified event frames."
    )
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--display-delay-ms", default=1, type=int)
    return load_args(parser)


if __name__ == "__main__":
    main()
