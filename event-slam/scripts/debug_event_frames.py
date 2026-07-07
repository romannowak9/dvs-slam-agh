#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
    EventFrameAggregator,
    EventFrameMode,
    PolarityMode,
)
from event_slam.events.event_filter import BackgroundActivityFilter
from event_slam.events.event_window import StereoEventWindowBuilder


def main() -> None:
    args = parse_args()

    image_shape = (args.height, args.width)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        )
        right_baf = BackgroundActivityFilter(
            image_shape=image_shape,
            time_window=args.baf_time_window,
        )

    saved_frames = 0

    for frame_index, window in enumerate(builder.iter_windows()):
        if saved_frames >= args.num_frames:
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

        left_path = output_dir / f"frame_{frame_index:05d}_left.png"
        right_path = output_dir / f"frame_{frame_index:05d}_right.png"

        save_image(left_path, stereo_frame.left.image)
        save_image(right_path, stereo_frame.right.image)

        if args.save_preview:
            preview = np.concatenate(
                (stereo_frame.left.image, stereo_frame.right.image),
                axis=1,
            )
            preview_path = output_dir / f"frame_{frame_index:05d}_preview.png"
            save_image(preview_path, preview)

        print(
            f"saved frame {frame_index:05d}: "
            f"t=[{window.t_start:.9f}, {window.t_end:.9f}), "
            f"left_events={len(window.left)}/{raw_left_count}, "
            f"right_events={len(window.right)}/{raw_right_count}"
        )

        saved_frames += 1

    print()
    print(f"saved_frames: {saved_frames}")
    print(f"output_dir: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create debug event frames from an EvSLAM ROS bag."
    )

    parser.add_argument(
        "--bag",
        required=True,
        type=Path,
        help="Path to the EvSLAM .bag file.",
    )

    parser.add_argument(
        "--left-topic",
        default=DEFAULT_LEFT_EVENT_TOPIC,
        help="Left event topic.",
    )

    parser.add_argument(
        "--right-topic",
        default=DEFAULT_RIGHT_EVENT_TOPIC,
        help="Right event topic.",
    )

    parser.add_argument(
        "--time-window",
        default=0.0333333333,
        type=float,
        help="Event window duration in seconds.",
    )

    parser.add_argument(
        "--num-frames",
        default=20,
        type=int,
        help="Number of stereo event frames to save.",
    )

    parser.add_argument(
        "--height",
        default=480,
        type=int,
        help="Image height.",
    )

    parser.add_argument(
        "--width",
        default=640,
        type=int,
        help="Image width.",
    )

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

    parser.add_argument(
        "--tau",
        default=0.03,
        type=float,
        help="Time constant for exponential decay frames.",
    )

    parser.add_argument(
        "--use-baf",
        action="store_true",
        help="Enable Background Activity Filter before aggregation.",
    )

    parser.add_argument(
        "--baf-time-window",
        default=1.0 / 24.0,
        type=float,
        help="BAF temporal neighborhood in seconds.",
    )

    parser.add_argument(
        "--t-start",
        default=None,
        type=float,
        help="Optional processing start time in seconds.",
    )

    parser.add_argument(
        "--t-end",
        default=None,
        type=float,
        help="Optional processing end time in seconds.",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/debug_frames",
        type=Path,
        help="Directory for saved debug frames.",
    )

    parser.add_argument(
        "--save-preview",
        action="store_true",
        help="Save side-by-side left/right preview images.",
    )

    return parser.parse_args()


def save_image(path: Path, image: np.ndarray) -> None:
    ok = cv2.imwrite(str(path), image)

    if not ok:
        raise IOError(f"Could not save image: {path}")


if __name__ == "__main__":
    main()