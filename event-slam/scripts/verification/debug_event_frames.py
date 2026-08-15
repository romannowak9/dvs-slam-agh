#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"

if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))


from event_slam.datasets.evslam_reader import EvSlamRosbagReader
from event_slam.events.event_aggregator import EventFrameAggregator
from event_slam.events.event_filter import StereoBackgroundActivityFilter
from event_slam.events.event_window import StereoEventWindowBuilder

from event_slam.debug.visualization import (
    colorize_event_frame,
    make_side_by_side,
    save_image,
    show_image,
)
from verification_config import load_args, verification_parser

def main() -> None:
    _, args = parse_args()

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

    background_filter = None

    if args.use_baf:
        background_filter = StereoBackgroundActivityFilter(
            image_shape=image_shape,
            time_window=args.baf_time_window,
            radius=args.baf_radius,
            min_neighbors=args.baf_min_neighbors,
        )

    processed_frames = 0

    for frame_index, window in enumerate(builder.iter_windows()):
        if args.num_frames > 0 and frame_index >= args.num_frames:
            break

        raw_left_count = len(window.left)
        raw_right_count = len(window.right)

        if args.use_baf:
            window = background_filter.filter(window)

        stereo_frame = aggregator.aggregate_stereo_window(window)

        left_color = colorize_event_frame(stereo_frame.left.image)
        right_color = colorize_event_frame(stereo_frame.right.image)
        preview = make_side_by_side(left_color, right_color)

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


def parse_args() -> tuple:
    parser = verification_parser(
        "Create debug event frames from an EvSLAM ROS bag."
    )
    parser.add_argument("--height", default=480, type=int)
    parser.add_argument("--width", default=640, type=int)
    parser.add_argument("--save-preview", action="store_true")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--display-delay-ms", default=1, type=int)
    return load_args(parser)


if __name__ == "__main__":
    main()
