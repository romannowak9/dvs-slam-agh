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
from event_slam.vo.feature_tracker import FeatureDetectorMode, FeatureTracker


def main() -> None:
    args = parse_args()
    cv2 = import_cv2()

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
        if processed >= args.num_frames:
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
        draw_tracks(cv2, debug_image, result)
        draw_stats(cv2, debug_image, result)

        if args.display:
            keep_running = show_image(
                cv2=cv2,
                window_name="debug_feature_tracking",
                image=debug_image,
                delay_ms=args.display_delay_ms,
            )

            if not keep_running:
                break
        else:
            output_path = output_dir / f"frame_{frame_index:05d}.png"
            save_image(cv2, output_path, debug_image)

        print(
            f"frame {frame_index:05d}: "
            f"tracks={result.track_count}, "
            f"active={result.active_count}, "
            f"redetected={result.redetected}, "
            f"detected={result.detected_count}, "
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
    parser.add_argument("--num-frames", default=100, type=int)

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


def colorize_event_frame(gray: np.ndarray) -> np.ndarray:
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


def draw_tracks(cv2, image: np.ndarray, result) -> None:
    if result.track_count > 0:
        for prev_pt, curr_pt in zip(result.prev_points, result.curr_points):
            x0, y0 = int(round(prev_pt[0])), int(round(prev_pt[1]))
            x1, y1 = int(round(curr_pt[0])), int(round(curr_pt[1]))

            cv2.line(image, (x0, y0), (x1, y1), (0, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(image, (x1, y1), 2, (0, 255, 255), -1, cv2.LINE_AA)
    else:
        for point in result.active_points:
            x, y = int(round(point[0])), int(round(point[1]))
            cv2.circle(image, (x, y), 2, (0, 255, 255), -1, cv2.LINE_AA)


def draw_stats(cv2, image: np.ndarray, result) -> None:
    text = (
        f"tracks={result.track_count} "
        f"active={result.active_count} "
        f"redetect={int(result.redetected)}"
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