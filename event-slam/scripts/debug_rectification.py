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


def main() -> None:
    args = parse_args()
    cv2 = import_cv2()

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
        polarity_mode=PolarityMode.BOTH,
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

    processed = 0

    for frame_index, window in enumerate(builder.iter_windows()):
        if processed >= args.num_frames:
            break

        if args.use_baf:
            window = StereoEventWindow(
                t_start=window.t_start,
                t_end=window.t_end,
                left=left_baf.filter(window.left),
                right=right_baf.filter(window.right),
            )

        stereo_frame = aggregator.aggregate_stereo_window(window)

        raw_left = colorize_event_frame(stereo_frame.left.image)
        raw_right = colorize_event_frame(stereo_frame.right.image)

        rect_left_gray, rect_right_gray = rectifier.rectify_pair(
            stereo_frame.left.image,
            stereo_frame.right.image,
        )

        rect_left = colorize_event_frame(rect_left_gray)
        rect_right = colorize_event_frame(rect_right_gray)

        raw_pair = make_pair(raw_left, raw_right)
        rectified_pair = make_pair(rect_left, rect_right)

        if args.draw_lines:
            lines_pair = draw_horizontal_lines(
                rectified_pair.copy(),
                step=args.line_step,
            )

        if args.display:
            shown = lines_pair if args.draw_lines else rectified_pair
            keep_running = show_image(
                cv2=cv2,
                window_name="debug_rectification",
                image=shown,
                delay_ms=args.display_delay_ms,
            )

            if not keep_running:
                break
        else:
            save_image(cv2, raw_dir / f"frame_{frame_index:05d}.png", raw_pair)
            save_image(
                cv2,
                rectified_dir / f"frame_{frame_index:05d}.png",
                rectified_pair,
            )

            if args.draw_lines:
                save_image(
                    cv2,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug stereo rectification on EvSLAM event frames."
    )

    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument("--camera-yaml", required=True, type=Path)

    parser.add_argument("--left-topic", default=DEFAULT_LEFT_EVENT_TOPIC)
    parser.add_argument("--right-topic", default=DEFAULT_RIGHT_EVENT_TOPIC)

    parser.add_argument("--time-window", default=0.0333333333, type=float)
    parser.add_argument("--num-frames", default=20, type=int)

    parser.add_argument(
        "--mode",
        default=EventFrameMode.STANDARD.value,
        choices=[mode.value for mode in EventFrameMode],
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
        "--interpolation",
        default="nearest",
        choices=["nearest", "linear"],
    )

    parser.add_argument("--draw-lines", action="store_true")
    parser.add_argument("--line-step", default=40, type=int)

    parser.add_argument("--output-dir", default="outputs/debug_rectification", type=Path)

    parser.add_argument("--display", action="store_true")
    parser.add_argument("--display-delay-ms", default=1, type=int)

    return parser.parse_args()


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


def colorize_event_frame(gray: np.ndarray) -> np.ndarray:
    """
    Convert a grayscale event frame to a BGR visualization.

    Output:
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
    color[:, :, 0] = positive_strength
    color[:, :, 2] = negative_strength

    return color


def make_pair(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.concatenate((left, right), axis=1)


def draw_horizontal_lines(image: np.ndarray, step: int) -> np.ndarray:
    cv2 = import_cv2()

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