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

    calibration = load_stereo_calibration(args.camera_yaml)
    image_shape = calibration.left.image_shape

    rectifier = StereoRectifier(
        calibration=calibration,
        image_shape=image_shape,
        alpha=args.alpha,
        interpolation=args.interpolation,
    )

    print_rectification_info(rectifier)

    if not args.display:
        output_dir = Path(args.output_dir)
        raw_dir = output_dir / "raw"
        rectified_dir = output_dir / "rectified"
        lines_dir = output_dir / "lines"

        raw_dir.mkdir(parents=True, exist_ok=True)
        rectified_dir.mkdir(parents=True, exist_ok=True)

        if args.draw_lines:
            lines_dir.mkdir(parents=True, exist_ok=True)
    else:
        raw_dir = None
        rectified_dir = None
        lines_dir = None

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

    processed = 0

    for frame_index, window in enumerate(builder.iter_windows()):
        if args.num_frames > 0 and processed >= args.num_frames:
            break

        if args.use_baf:
            window = background_filter.filter(window)

        stereo_frame = aggregator.aggregate_stereo_window(window)

        raw_left = colorize_event_frame(stereo_frame.left.image)
        raw_right = colorize_event_frame(stereo_frame.right.image)

        rect_left_gray, rect_right_gray = rectifier.rectify_pair(
            stereo_frame.left.image,
            stereo_frame.right.image,
        )

        rect_left = colorize_event_frame(rect_left_gray)
        rect_right = colorize_event_frame(rect_right_gray)

        raw_pair = make_side_by_side(raw_left, raw_right)
        rectified_pair = make_side_by_side(rect_left, rect_right)

        if args.draw_lines:
            lines_pair = draw_horizontal_lines(
                rectified_pair.copy(),
                step=args.line_step,
            )

        if args.display:
            shown = lines_pair if args.draw_lines else rectified_pair
            keep_running = show_image(
                window_name="debug_rectification",
                image=shown,
                delay_ms=args.display_delay_ms,
            )

            if not keep_running:
                break
        else:
            save_image(raw_dir / f"frame_{frame_index:05d}.png", raw_pair)
            save_image(
                            rectified_dir / f"frame_{frame_index:05d}.png",
                rectified_pair,
            )

            if args.draw_lines:
                save_image(
                                    lines_dir / f"frame_{frame_index:05d}.png",
                    lines_pair,
                )

        print(
            f"frame {frame_index:05d}: "
            f"t=[{window.t_start:.9f}, {window.t_end:.9f}), "
            f"left_events={len(window.left)}, right_events={len(window.right)}"
        )

        processed += 1

    if args.display:
        cv2.destroyAllWindows()

    print()
    print(f"processed_frames: {processed}")

    if not args.display:
        print(f"output_dir: {Path(args.output_dir)}")


def print_rectification_info(rectifier: StereoRectifier) -> None:
    print()
    print("=" * 80)
    print("Stereo rectification")
    print("=" * 80)
    print(f"baseline_rectified [m]: {rectifier.baseline:.9f}")
    print("K_left_rectified:")
    print(np.array2string(rectifier.K_left_rectified, precision=6, suppress_small=False))
    print("K_right_rectified:")
    print(np.array2string(rectifier.K_right_rectified, precision=6, suppress_small=False))
    print("P1:")
    print(np.array2string(rectifier.P1, precision=6, suppress_small=False))
    print("P2:")
    print(np.array2string(rectifier.P2, precision=6, suppress_small=False))


def draw_horizontal_lines(image: np.ndarray, step: int) -> np.ndarray:

    if step <= 0:
        return image

    for y in range(0, image.shape[0], step):
        cv2.line(
            image,
            (0, y),
            (image.shape[1] - 1, y),
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return image


def parse_args() -> tuple:
    parser = verification_parser(
        "Debug stereo rectification on EvSLAM event frames."
    )
    parser.add_argument("--draw-lines", action="store_true")
    parser.add_argument("--line-step", default=40, type=int)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--display-delay-ms", default=1, type=int)
    return load_args(parser)


if __name__ == "__main__":
    main()
