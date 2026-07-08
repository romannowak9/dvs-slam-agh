#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


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
    BACKGROUND_INTENSITY,
    EventFrameAggregator,
    EventFrameMode,
    PolarityMode,
)
from event_slam.events.event_filter import BackgroundActivityFilter
from event_slam.events.event_window import StereoEventWindowBuilder


def main() -> None:
    args = parse_args()
    cv2 = import_cv2()

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
        if processed_frames >= args.num_frames:
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
                cv2=cv2,
                window_name="debug_event_frames",
                image=preview,
                delay_ms=args.display_delay_ms,
            )

            if not keep_running:
                break
        else:
            left_path = left_output_dir / f"frame_{frame_index:05d}.png"
            right_path = right_output_dir / f"frame_{frame_index:05d}.png"

            save_image(cv2, left_path, left_color)
            save_image(cv2, right_path, right_color)

            if args.save_preview:
                preview_path = both_output_dir / f"frame_{frame_index:05d}.png"
                save_image(cv2, preview_path, preview)

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
    parser.add_argument("--num-frames", default=20, type=int)
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


def colorize_event_frame(gray: np.ndarray) -> np.ndarray:
    """
    Convert a grayscale event frame to a BGR visualization.

    Input convention:
        127 -> background
        >127 -> positive events
        <127 -> negative events

    Output convention:
        black background
        blue positive events
        red negative events
    """
    if gray.ndim != 2:
        raise ValueError(f"Expected a single-channel image, got shape {gray.shape}")

    gray_i16 = gray.astype(np.int16)

    positive = gray_i16 > BACKGROUND_INTENSITY
    negative = gray_i16 < BACKGROUND_INTENSITY

    positive_strength = np.zeros_like(gray, dtype=np.uint8)
    negative_strength = np.zeros_like(gray, dtype=np.uint8)

    positive_strength[positive] = np.clip(
        2 * (gray_i16[positive] - BACKGROUND_INTENSITY),
        0,
        255,
    ).astype(np.uint8)

    negative_strength[negative] = np.clip(
        2 * (BACKGROUND_INTENSITY - gray_i16[negative]),
        0,
        255,
    ).astype(np.uint8)

    color = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)

    color[:, :, 0] = positive_strength  # blue channel in OpenCV BGR
    color[:, :, 2] = negative_strength  # red channel in OpenCV BGR

    return color


def show_image(cv2, window_name: str, image: np.ndarray, delay_ms: int) -> bool:
    cv2.imshow(window_name, image)

    key = cv2.waitKey(max(0, int(delay_ms))) & 0xFF

    if key in (27, ord("q")):
        return False

    return True


def save_image(cv2, path: Path, image: np.ndarray) -> None:
    ok = cv2.imwrite(str(path), image)

    if not ok:
        raise IOError(f"Could not save image: {path}")


def import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV Python is required. Install it with: apt install python3-opencv"
        ) from exc

    return cv2


if __name__ == "__main__":
    main()