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


DEFAULT_COLOR_TOPIC = "/camera/color/image_raw/compressed"
DEFAULT_DEPTH_TOPIC = "/camera/depth/image_rect_raw/compressed"
DEFAULT_INFRA_TOPIC = "/camera/infra1/image_rect_raw/compressed"


def main() -> None:
    args = parse_args()

    image_shape = (args.height, args.width)

    if not args.display:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None

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

    if args.use_baf:
        left_baf = BackgroundActivityFilter(
            image_shape=image_shape,
            time_window=args.baf_time_window,
            radius=args.baf_radius,
            min_neighbors=args.baf_min_neighbors,
        )

    read_start = None

    if args.t_start is not None:
        read_start = max(0.0, args.t_start - args.match_margin)

    event_stream = LeftEventFrameStream(
        builder=builder,
        aggregator=aggregator,
        baf_filter=left_baf,
    )

    color_stream = CompressedImageStream(
        reader=reader,
        topic=args.color_topic,
        image_kind="color",
        start_time=read_start,
    )

    depth_stream = None
    infra_stream = None

    if not args.only_color:
        depth_stream = CompressedImageStream(
            reader=reader,
            topic=args.depth_topic,
            image_kind="depth",
            start_time=read_start,
        )

        infra_stream = CompressedImageStream(
            reader=reader,
            topic=args.infra_topic,
            image_kind="infra",
            start_time=read_start,
        )

    processed = 0

    while processed < args.num_frames and args.num_frames == 0:
        color_item = color_stream.get_next()

        if color_item is None:
            break

        color_time, color_image = color_item

        if args.t_start is not None and color_time < args.t_start:
            continue

        if args.t_end is not None and color_time > args.t_end:
            break

        event_image, event_time = event_stream.get_nearest(color_time)

        if event_image is None:
            print("No matching event frame found. Stopping.")
            break

        if args.only_color:
            compare = make_only_color_layout(
                color_image=color_image,
                event_image=event_image,
                image_shape=image_shape,
                color_time=color_time,
                event_time=event_time,
            )
        else:
            depth_image, depth_time = depth_stream.get_nearest(color_time)
            infra_image, infra_time = infra_stream.get_nearest(color_time)

            compare = make_full_layout(
                color_image=color_image,
                depth_image=depth_image,
                event_image=event_image,
                infra_image=infra_image,
                image_shape=image_shape,
                color_time=color_time,
                depth_time=depth_time,
                event_time=event_time,
                infra_time=infra_time,
            )

        if args.display:
            keep_running = show_image(
                window_name="compare_camera_frames",
                image=compare,
                delay_ms=args.display_delay_ms,
            )

            if not keep_running:
                break
        else:
            output_path = output_dir / f"compare_{processed:05d}.png"
            save_image(output_path, compare)

        print(
            f"frame {processed:05d}: "
            f"color_t={color_time:.9f}, "
            f"event_dt={format_dt(event_time, color_time)}"
        )

        processed += 1

    if args.display:
        cv2.destroyAllWindows()

    print()
    print(f"processed_frames: {processed}")

    if args.display:
        print("display_mode: True")
    else:
        print(f"output_dir: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare event frames with regular camera frames from EvSLAM bag."
    )

    parser.add_argument("--bag", required=True, type=Path)

    parser.add_argument("--left-topic", default=DEFAULT_LEFT_EVENT_TOPIC)
    parser.add_argument("--right-topic", default=DEFAULT_RIGHT_EVENT_TOPIC)

    parser.add_argument("--color-topic", default=DEFAULT_COLOR_TOPIC)
    parser.add_argument("--depth-topic", default=DEFAULT_DEPTH_TOPIC)
    parser.add_argument("--infra-topic", default=DEFAULT_INFRA_TOPIC)

    parser.add_argument("--time-window", default=0.0333333333, type=float)
    parser.add_argument("--num-frames", default=20, type=int, help="Number of frames to process. Use 0 to process the whole bag.",)
    parser.add_argument("--height", default=480, type=int)
    parser.add_argument("--width", default=640, type=int)

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

    parser.add_argument(
        "--match-margin",
        default=0.1,
        type=float,
        help="How many seconds before t_start should image streams start reading.",
    )

    parser.add_argument(
        "--only-color",
        action="store_true",
        help="Compare only color camera with left event frame.",
    )

    parser.add_argument("--output-dir", default="outputs/camera_compare", type=Path)

    parser.add_argument(
        "--display",
        action="store_true",
        help="Display comparison frames instead of saving images.",
    )

    parser.add_argument(
        "--display-delay-ms",
        default=1,
        type=int,
        help="Delay for cv2.waitKey in display mode. Use 0 to step frame by frame.",
    )

    return parser.parse_args()


class LeftEventFrameStream:
    """
    Streaming nearest-neighbor access to generated left event frames.

    The stream keeps only previous and next event frame in memory.
    """

    def __init__(
        self,
        builder: StereoEventWindowBuilder,
        aggregator: EventFrameAggregator,
        baf_filter: BackgroundActivityFilter | None = None,
    ) -> None:
        self.iterator = builder.iter_windows()
        self.aggregator = aggregator
        self.baf_filter = baf_filter

        self.previous = None
        self.next_item = None
        self.exhausted = False

        self._advance()

    def get_nearest(self, timestamp: float):
        while self.next_item is not None and self.next_item[0] <= timestamp:
            self.previous = self.next_item
            self._advance()

        candidates = []

        if self.previous is not None:
            candidates.append(self.previous)

        if self.next_item is not None:
            candidates.append(self.next_item)

        if not candidates:
            return None, None

        nearest = min(candidates, key=lambda item: abs(item[0] - timestamp))
        return nearest[1], nearest[0]

    def _advance(self) -> None:
        if self.exhausted:
            self.next_item = None
            return

        try:
            window = next(self.iterator)
        except StopIteration:
            self.exhausted = True
            self.next_item = None
            return

        if self.baf_filter is not None:
            left_batch = self.baf_filter.filter(window.left)

            window = StereoEventWindow(
                t_start=window.t_start,
                t_end=window.t_end,
                left=left_batch,
                right=window.right,
            )

        left_frame = self.aggregator.aggregate_batch(
            batch=window.left,
            t_start=window.t_start,
            t_end=window.t_end,
        )

        image = colorize_event_frame(left_frame.image)
        timestamp = 0.5 * (window.t_start + window.t_end)

        self.next_item = (timestamp, image)


class CompressedImageStream:
    """
    Streaming image reader with nearest-neighbor timestamp matching.
    """

    def __init__(
        self,
        reader: EvSlamRosbagReader,
        topic: str,
        image_kind: str,
        start_time: float | None = None,
    ) -> None:
        self.iterator = reader.iter_messages(topics=topic, start_time=start_time)
        self.image_kind = image_kind

        self.previous = None
        self.next_item = None
        self.exhausted = False

        self._advance()

    def get_next(self):
        if self.next_item is None:
            return None

        item = self.next_item
        self.previous = item
        self._advance()

        return item

    def get_nearest(self, timestamp: float):
        while self.next_item is not None and self.next_item[0] <= timestamp:
            self.previous = self.next_item
            self._advance()

        candidates = []

        if self.previous is not None:
            candidates.append(self.previous)

        if self.next_item is not None:
            candidates.append(self.next_item)

        if not candidates:
            return None, None

        nearest = min(candidates, key=lambda item: abs(item[0] - timestamp))
        return nearest[1], nearest[0]

    def _advance(self) -> None:
        if self.exhausted:
            self.next_item = None
            return

        try:
            _, msg, bag_time = next(self.iterator)
        except StopIteration:
            self.exhausted = True
            self.next_item = None
            return

        timestamp = get_message_timestamp(msg, bag_time)
        raw_image = decode_compressed_image(msg)
        image = visualize_sensor_image(raw_image, self.image_kind)

        self.next_item = (timestamp, image)


def decode_compressed_image(msg) -> np.ndarray:
    data = np.frombuffer(msg.data, dtype=np.uint8)

    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

    if image is not None:
        return image

    for offset in (12, 16):
        if data.size > offset:
            image = cv2.imdecode(data[offset:], cv2.IMREAD_UNCHANGED)

            if image is not None:
                return image

    raise ValueError("Could not decode compressed image")


def visualize_sensor_image(image: np.ndarray, image_kind: str) -> np.ndarray:
    if image_kind == "depth":
        return visualize_depth(image)

    if image_kind in ("color", "infra"):
        return to_bgr(normalize_to_uint8(image))

    raise ValueError(f"Unsupported image kind: {image_kind}")


def visualize_depth(image: np.ndarray) -> np.ndarray:
    depth_u8 = normalize_to_uint8(image, ignore_zero=True)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)


def normalize_to_uint8(
    image: np.ndarray,
    ignore_zero: bool = False,
) -> np.ndarray:
    if image.dtype == np.uint8:
        return image

    image_f = image.astype(np.float64)

    if ignore_zero:
        valid = np.isfinite(image_f) & (image_f > 0)
    else:
        valid = np.isfinite(image_f)

    if not np.any(valid):
        return np.zeros(image.shape[:2], dtype=np.uint8)

    low = np.percentile(image_f[valid], 1.0)
    high = np.percentile(image_f[valid], 99.0)

    if high <= low:
        return np.zeros(image.shape[:2], dtype=np.uint8)

    normalized = (image_f - low) / (high - low)
    normalized = np.clip(normalized, 0.0, 1.0)

    return np.rint(255.0 * normalized).astype(np.uint8)


def to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    if image.ndim == 3 and image.shape[2] == 3:
        return image

    raise ValueError(f"Unsupported image shape: {image.shape}")


def make_full_layout(
    color_image: np.ndarray,
    depth_image: np.ndarray | None,
    event_image: np.ndarray,
    infra_image: np.ndarray | None,
    image_shape: tuple,
    color_time: float,
    depth_time: float | None,
    event_time: float,
    infra_time: float | None,
) -> np.ndarray:
    color_image = prepare_tile(color_image, image_shape)
    depth_image = prepare_tile(depth_image, image_shape)
    event_image = prepare_tile(event_image, image_shape)
    infra_image = prepare_tile(infra_image, image_shape)

    draw_label(color_image, f"color  t={color_time:.6f}")
    draw_label(depth_image, f"depth  dt={format_dt(depth_time, color_time)}")
    draw_label(event_image, f"left events  dt={format_dt(event_time, color_time)}")
    draw_label(infra_image, f"infra  dt={format_dt(infra_time, color_time)}")

    top = np.concatenate((color_image, depth_image), axis=1)
    bottom = np.concatenate((event_image, infra_image), axis=1)

    return np.concatenate((top, bottom), axis=0)


def make_only_color_layout(
    color_image: np.ndarray,
    event_image: np.ndarray,
    image_shape: tuple,
    color_time: float,
    event_time: float,
) -> np.ndarray:
    color_image = prepare_tile(color_image, image_shape)
    event_image = prepare_tile(event_image, image_shape)

    draw_label(color_image, f"color  t={color_time:.6f}")
    draw_label(event_image, f"left events  dt={format_dt(event_time, color_time)}")

    return np.concatenate((color_image, event_image), axis=0)


def prepare_tile(
    image: np.ndarray | None,
    image_shape: tuple,
) -> np.ndarray:
    height, width = int(image_shape[0]), int(image_shape[1])

    if image is None:
        return np.zeros((height, width, 3), dtype=np.uint8)

    image = to_bgr(image)

    if image.shape[0] != height or image.shape[1] != width:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    return image.copy()


def draw_label(image: np.ndarray, text: str) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(
        image,
        text,
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def get_message_timestamp(msg, bag_time) -> float:
    if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
        stamp = msg.header.stamp

        if hasattr(stamp, "to_sec"):
            return float(stamp.to_sec())

        if hasattr(stamp, "secs") and hasattr(stamp, "nsecs"):
            return float(stamp.secs) + float(stamp.nsecs) * 1e-9

    if hasattr(bag_time, "to_sec"):
        return float(bag_time.to_sec())

    return float(bag_time)


def format_dt(time_value: float | None, reference_time: float) -> str:
    if time_value is None:
        return "None"

    return f"{time_value - reference_time:+.6f}s"


if __name__ == "__main__":
    main()