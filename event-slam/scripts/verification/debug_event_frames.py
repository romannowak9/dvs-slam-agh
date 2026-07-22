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

from event_slam.debug.visualization import (
    colorize_event_frame,
    save_image,
    show_image,
)

def main() -> None:
    args = parse_args()

    image_shape = (args.height, args.width)

    if not args.display:
        output_dir = Path(args.output_dir)

        left_output_dir = output_dir / "left"
        right_output_dir = output_dir / "right"
        both_output_dir = output_dir / "both"

        left_output_dir.mkdir(parents=True, exist_ok=True)
        right_output_dir.mkdir(parents=True, exist_ok=True)

        if args.save_preview:
            both_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None
        left_output_dir = None
        right_output_dir = None
        both_output_dir = None

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

    processed_frames = 0

    for frame_index, window in enumerate(builder.iter_windows()):
        if args.num_frames > 0 and processed_frames >= args.num_frames:
            break

        raw_left_count = len(window.left)
        raw_right_count = len(window.right)

        if args.use_baf:
            left_batch = left_baf.filter(window.left)
            right_batch = right_baf.filter(window.right)

            window = StereoEventWindow(
                t_start=window.t_start,
                t_end=window.t_end,
                left=left_batch,
                right=right_batch,
            )

        stereo_frame = aggregator.aggregate_stereo_window(window)

        left_color = colorize_event_frame(stereo_frame.left.image)
        right_color = colorize_event_frame(stereo_frame.right.image)
        preview = np.concatenate((left_color, right_color), axis=1)

        if args.display:
            keep_running = show_image(
                window_name="debug_event_frames",
                image=preview,
                delay_ms=args.display_delay_ms,
            )

            if not keep_running:
                break
        else:
            left_path = left_output_dir / f"frame_{frame_index:05d}.png"
            right_path = right_output_dir / f"frame_{frame_index:05d}.png"

            save_image(left_path, left_color)
            save_image(right_path, right_color)

            if args.save_preview:
                preview_path = both_output_dir / f"frame_{frame_index:05d}.png"
                save_image(preview_path, preview)

        print(
            f"frame {frame_index:05d}: "
            f"t=[{window.t_start:.9f}, {window.t_end:.9f}), "
            f"left_events={len(window.left)}/{raw_left_count}, "
            f"right_events={len(window.right)}/{raw_right_count}"
        )

        processed_frames += 1

    if args.display:
        cv2.destroyAllWindows()

    print()
    print(f"processed_frames: {processed_frames}")

    if args.display:
        print("display_mode: True")
    else:
        print(f"output_dir: {output_dir}")

    if args.use_baf:
        print(
            "baf: "
            f"time_window={args.baf_time_window}, "
            f"radius={args.baf_radius}, "
            f"min_neighbors={args.baf_min_neighbors}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create debug event frames from an EvSLAM ROS bag."
    )

    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument("--left-topic", default=DEFAULT_LEFT_EVENT_TOPIC)
    parser.add_argument("--right-topic", default=DEFAULT_RIGHT_EVENT_TOPIC)

    parser.add_argument("--time-window", default=0.0333333333, type=float)
    parser.add_argument("--num-frames", default=20, type=int, help="Number of frames to process. Use 0 to process the whole bag.",)
    parser.add_argument("--height", default=480, type=int)
    parser.add_argument("--width", default=640, type=int)

    parser.add_argument(
        "--mode",
        default=EventFrameMode.STANDARD.value,
        choices=[mode.value for mode in EventFrameMode],
        help="Event frame aggregation mode.",
    )

    parser.add_argument(
        "--polarity-mode",
        default=PolarityMode.BOTH.value,
        choices=[mode.value for mode in PolarityMode],
        help="Which event polarities should be rendered.",
    )

    parser.add_argument("--tau", default=0.03, type=float)

    parser.add_argument("--use-baf", action="store_true")
    parser.add_argument("--baf-time-window", default=1.0 / 24.0, type=float)
    parser.add_argument("--baf-radius", default=2, type=int)
    parser.add_argument("--baf-min-neighbors", default=1, type=int)

    parser.add_argument("--t-start", default=None, type=float)
    parser.add_argument("--t-end", default=None, type=float)

    parser.add_argument("--output-dir", default="outputs/debug_frames", type=Path)

    parser.add_argument(
        "--save-preview",
        action="store_true",
        help="Save side-by-side left/right preview images.",
    )

    parser.add_argument(
        "--display",
        action="store_true",
        help="Display side-by-side event frames instead of saving images.",
    )

    parser.add_argument(
        "--display-delay-ms",
        default=1,
        type=int,
        help="Delay for cv2.waitKey in display mode. Use 0 to step frame by frame.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()