#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"

if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))


from event_slam.calibration.kalibr_parser import (
    load_imu_calibration,
    load_stereo_calibration,
)
from event_slam.core.imu import ImuCoverageError
from event_slam.datasets.evslam_reader import (
    DEFAULT_LEFT_EVENT_TOPIC,
    DEFAULT_RIGHT_EVENT_TOPIC,
    EvSlamRosbagReader,
)
from event_slam.debug.visualization import (
    colorize_event_frame,
    save_image,
    show_image,
)
from event_slam.events.event_aggregator import (
    EventFrameAggregator,
    EventFrameMode,
    PolarityMode,
)
from event_slam.events.event_filter import StereoBackgroundActivityFilter
from event_slam.events.event_window import StereoEventWindowBuilder
from event_slam.events.imu_motion_compensation import (
    compensate_stereo_window_rotation,
)


def main() -> None:
    args = parse_args()

    stereo_calibration = load_stereo_calibration(args.camera_calibration)
    imu_calibration = load_imu_calibration(args.imu_calibration)

    imu_topic = args.imu_topic or imu_calibration.topic

    if imu_topic is None:
        raise ValueError(
            "IMU topic was not provided and could not be read from IMU calibration"
        )

    image_shape = (args.height, args.width)
    output_paths = setup_output_paths(args)

    reader = EvSlamRosbagReader(
        bag_path=args.bag,
        left_event_topic=args.left_topic,
        right_event_topic=args.right_topic,
    )

    imu_timestamps, angular_velocities = reader.load_imu_gyro(topic=imu_topic)

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

        try:
            compensation_result = compensate_stereo_window_rotation(
                window=window,
                left_camera=stereo_calibration.left,
                right_camera=stereo_calibration.right,
                imu_timestamps=imu_timestamps,
                angular_velocities=angular_velocities,
                T_C_left_imu=stereo_calibration.T_C_left_imu,
                T_C_right_imu=stereo_calibration.T_C_right_imu,
                timeshift_left_imu=stereo_calibration.timeshift_left_imu
                + args.extra_timeshift,
                timeshift_right_imu=stereo_calibration.timeshift_right_imu
                + args.extra_timeshift,
                imu_time_offset=imu_calibration.time_offset or 0.0,
                reference_time=args.reference_time,
                num_time_bins=args.time_bins,
            )
        except ImuCoverageError as exc:
            print(f"frame {frame_index:05d}: skipped, {exc}")
            continue

        raw_frame = aggregator.aggregate_stereo_window(window)
        compensated_frame = aggregator.aggregate_stereo_window(
            compensation_result.window
        )

        left_raw = colorize_event_frame(raw_frame.left.image)
        right_raw = colorize_event_frame(raw_frame.right.image)
        left_comp = colorize_event_frame(compensated_frame.left.image)
        right_comp = colorize_event_frame(compensated_frame.right.image)

        preview = make_preview(
            left_raw=left_raw,
            left_comp=left_comp,
            right_raw=right_raw,
            right_comp=right_comp,
        )

        if args.display:
            keep_running = show_image(
                window_name="debug_imu_motion_compensation",
                image=preview,
                delay_ms=args.display_delay_ms,
            )

            if not keep_running:
                break
        else:
            save_debug_images(
                output_paths=output_paths,
                frame_index=frame_index,
                left_raw=left_raw,
                right_raw=right_raw,
                left_comp=left_comp,
                right_comp=right_comp,
                preview=preview,
                save_preview=args.save_preview,
            )

        print_frame_report(
            frame_index=frame_index,
            window=window,
            raw_left_count=raw_left_count,
            raw_right_count=raw_right_count,
            left_stats=compensation_result.left_stats,
            right_stats=compensation_result.right_stats,
        )

        processed_frames += 1

    if args.display:
        cv2.destroyAllWindows()

    print()
    print(f"processed_frames: {processed_frames}")

    if args.display:
        print("display_mode: True")
    else:
        print(f"output_dir: {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare event frames before and after IMU rotational motion "
            "compensation."
        )
    )

    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument("--camera-calibration", required=True, type=Path)
    parser.add_argument("--imu-calibration", required=True, type=Path)

    parser.add_argument("--left-topic", default=DEFAULT_LEFT_EVENT_TOPIC)
    parser.add_argument("--right-topic", default=DEFAULT_RIGHT_EVENT_TOPIC)
    parser.add_argument("--imu-topic", default=None, type=str)

    parser.add_argument("--time-window", default=0.0333333333, type=float)
    parser.add_argument(
        "--num-frames",
        default=20,
        type=int,
        help="Number of frames to process. Use 0 to process the whole bag.",
    )
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

    parser.add_argument(
        "--reference-time",
        default="middle",
        choices=["start", "middle", "end"],
    )
    parser.add_argument("--time-bins", default=32, type=int)
    parser.add_argument("--extra-timeshift", default=0.0, type=float)
    parser.add_argument("--use-baf", action="store_true")
    parser.add_argument("--baf-time-window", default=1.0 / 24.0, type=float)
    parser.add_argument("--baf-radius", default=2, type=int)
    parser.add_argument("--baf-min-neighbors", default=1, type=int)

    parser.add_argument("--t-start", default=None, type=float)
    parser.add_argument("--t-end", default=None, type=float)

    parser.add_argument(
        "--output-dir",
        default="outputs/debug_imu_motion_compensation",
        type=Path,
    )
    parser.add_argument(
        "--save-preview",
        action="store_true",
        help="Save side-by-side comparison images.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display comparison images instead of saving images.",
    )
    parser.add_argument(
        "--display-delay-ms",
        default=1,
        type=int,
        help="Delay for cv2.waitKey in display mode. Use 0 to step frame by frame.",
    )

    return parser.parse_args()


def setup_output_paths(args: argparse.Namespace) -> dict:
    if args.display:
        return {}

    output_dir = Path(args.output_dir)

    paths = {
        "left_raw": output_dir / "left_raw",
        "left_compensated": output_dir / "left_compensated",
        "right_raw": output_dir / "right_raw",
        "right_compensated": output_dir / "right_compensated",
        "preview": output_dir / "preview",
    }

    for key, path in paths.items():
        if key == "preview" and not args.save_preview:
            continue

        path.mkdir(parents=True, exist_ok=True)

    return paths


def make_preview(
    left_raw: np.ndarray,
    left_comp: np.ndarray,
    right_raw: np.ndarray,
    right_comp: np.ndarray,
) -> np.ndarray:
    left_pair = np.concatenate(
        (
            add_label(left_raw, "left raw"),
            add_label(left_comp, "left imu compensated"),
        ),
        axis=1,
    )
    right_pair = np.concatenate(
        (
            add_label(right_raw, "right raw"),
            add_label(right_comp, "right imu compensated"),
        ),
        axis=1,
    )

    return np.concatenate((left_pair, right_pair), axis=0)


def add_label(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()

    cv2.putText(
        output,
        text,
        org=(12, 24),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.6,
        color=(255, 255, 255),
        thickness=1,
        lineType=cv2.LINE_AA,
    )

    return output


def save_debug_images(
    output_paths: dict,
    frame_index: int,
    left_raw: np.ndarray,
    right_raw: np.ndarray,
    left_comp: np.ndarray,
    right_comp: np.ndarray,
    preview: np.ndarray,
    save_preview: bool,
) -> None:
    save_image(
        output_paths["left_raw"] / f"frame_{frame_index:05d}.png",
        left_raw,
    )
    save_image(
        output_paths["left_compensated"] / f"frame_{frame_index:05d}.png",
        left_comp,
    )
    save_image(
        output_paths["right_raw"] / f"frame_{frame_index:05d}.png",
        right_raw,
    )
    save_image(
        output_paths["right_compensated"] / f"frame_{frame_index:05d}.png",
        right_comp,
    )

    if save_preview:
        save_image(
            output_paths["preview"] / f"frame_{frame_index:05d}.png",
            preview,
        )


def print_frame_report(
    frame_index: int,
    window: StereoEventWindow,
    raw_left_count: int,
    raw_right_count: int,
    left_stats,
    right_stats,
) -> None:
    print(
        f"frame {frame_index:05d}: "
        f"t=[{window.t_start:.9f}, {window.t_end:.9f}), "
        f"left_events={left_stats.output_count}/{left_stats.input_count}/{raw_left_count}, "
        f"right_events={right_stats.output_count}/{right_stats.input_count}/{raw_right_count}, "
        f"left_dropped={left_stats.dropped_count}, "
        f"right_dropped={right_stats.dropped_count}, "
        f"reference_time={left_stats.reference_time:.9f}"
    )


if __name__ == "__main__":
    main()
