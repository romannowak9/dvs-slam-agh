#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path



PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"

if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))


from event_slam.core.types import CameraId
from event_slam.debug.visualization import print_section, print_vector
from event_slam.datasets.evslam_reader import EvSlamRosbagReader
from verification_config import load_args, verification_parser


def main() -> None:
    _, args = parse_args()

    reader = EvSlamRosbagReader(
        bag_path=args.bag,
        left_event_topic=args.left_topic,
        right_event_topic=args.right_topic,
        imu_topic=args.imu_topic,
    )

    print_bag_summary(reader)

    inspect_event_stream(
        reader=reader,
        camera=CameraId.LEFT,
        max_batches=args.num_batches,
        sample_events=args.sample_events,
    )

    inspect_event_stream(
        reader=reader,
        camera=CameraId.RIGHT,
        max_batches=args.num_batches,
        sample_events=args.sample_events,
    )

    if args.inspect_imu:
        inspect_imu_stream(
            reader=reader,
            max_samples=args.num_batches,
        )

    if args.full_scan:
        full_scan_event_topics(reader)


def print_bag_summary(reader: EvSlamRosbagReader) -> None:
    print_section("Bag summary")

    start_time, end_time = reader.get_time_range()

    print(f"bag_path: {reader.bag_path}")
    print(f"start_time [s]: {start_time:.9f}")
    print(f"end_time   [s]: {end_time:.9f}")
    print(f"duration   [s]: {reader.get_duration():.9f}")

    print_section("Topics")
    topics_summary = reader.get_topics_summary()

    for topic_name in sorted(topics_summary.keys()):
        info = topics_summary[topic_name]

        msg_type = info.get("msg_type")
        message_count = info.get("message_count")
        frequency = info.get("frequency")

        frequency_text = "None" if frequency is None else f"{frequency:.3f} Hz"

        print(
            f"{topic_name}: "
            f"type={msg_type}, "
            f"messages={message_count}, "
            f"frequency={frequency_text}"
        )


def inspect_event_stream(
    reader: EvSlamRosbagReader,
    camera: CameraId,
    max_batches: int,
    sample_events: int,
) -> None:
    print_section(f"Inspect {camera.value} event batches")

    total_batches = 0
    total_events = 0

    for batch_index, batch in enumerate(
        reader.iter_event_batches(
            camera=camera,
            max_batches=max_batches,
            include_empty_batches=True,
        )
    ):
        total_batches += 1
        total_events += len(batch)

        print()
        print(f"batch_index: {batch_index}")
        print(f"camera: {batch.camera.value}")
        print(f"event_count: {len(batch)}")

        if batch.time_range is None:
            print("time_range [s]: None")
        else:
            t_start, t_end = batch.time_range
            print(f"time_range [s]: {t_start:.9f} -> {t_end:.9f}")
            print(f"batch_duration [s]: {(t_end - t_start):.9f}")

        print_sample_events(batch, sample_events)

    print()
    print(f"inspected_batches: {total_batches}")
    print(f"inspected_events: {total_events}")


def print_sample_events(batch, sample_events: int) -> None:
    count = min(len(batch), sample_events)

    if count == 0:
        print("sample_events: []")
        return

    print("sample_events:")

    for index in range(count):
        print(
            f"  [{index}] "
            f"t={batch.t[index]:.9f}, "
            f"x={int(batch.x[index])}, "
            f"y={int(batch.y[index])}, "
            f"p={int(batch.p[index])}"
        )


def inspect_imu_stream(
    reader: EvSlamRosbagReader,
    max_samples: int,
) -> None:
    print_section("Inspect IMU samples")

    for sample_index, sample in enumerate(
        reader.iter_imu_samples(max_samples=max_samples)
    ):
        print()
        print(f"sample_index: {sample_index}")
        print(f"topic: {sample.topic}")
        print(f"frame_id: {sample.frame_id}")
        print(f"timestamp [s]: {sample.timestamp:.9f}")
        print_vector("orientation_xyzw", sample.orientation_xyzw)
        print_vector("angular_velocity", sample.angular_velocity)
        print_vector("linear_acceleration", sample.linear_acceleration)


def full_scan_event_topics(reader: EvSlamRosbagReader) -> None:
    print_section("Full event-topic scan")

    topics = [reader.left_event_topic, reader.right_event_topic]

    stats = {
        reader.left_event_topic: create_empty_event_stats(),
        reader.right_event_topic: create_empty_event_stats(),
    }

    for topic, msg, _ in reader.iter_messages(topics=topics):
        if not hasattr(msg, "events"):
            continue

        event_count = len(msg.events)
        topic_stats = stats[topic]

        topic_stats["message_count"] += 1
        topic_stats["event_count"] += event_count

        if event_count > 0:
            first_time = msg.events[0].ts.to_sec()
            last_time = msg.events[-1].ts.to_sec()

            update_time_range(topic_stats, first_time)
            update_time_range(topic_stats, last_time)

    for topic in topics:
        topic_stats = stats[topic]

        print()
        print(f"topic: {topic}")
        print(f"message_count: {topic_stats['message_count']}")
        print(f"event_count: {topic_stats['event_count']}")

        if topic_stats["t_min"] is None:
            print("event_time_range [s]: None")
        else:
            print(
                "event_time_range [s]: "
                f"{topic_stats['t_min']:.9f} -> {topic_stats['t_max']:.9f}"
            )


def create_empty_event_stats() -> dict:
    return {
        "message_count": 0,
        "event_count": 0,
        "t_min": None,
        "t_max": None,
    }


def update_time_range(stats: dict, timestamp: float) -> None:
    if stats["t_min"] is None or timestamp < stats["t_min"]:
        stats["t_min"] = timestamp

    if stats["t_max"] is None or timestamp > stats["t_max"]:
        stats["t_max"] = timestamp


def parse_args() -> tuple:
    parser = verification_parser(
        "Inspect an EvSLAM ROS bag without loading it into RAM."
    )
    parser.add_argument("--num-batches", default=3, type=int)
    parser.add_argument("--sample-events", default=5, type=int)
    parser.add_argument("--inspect-imu", action="store_true")
    parser.add_argument("--full-scan", action="store_true")
    return load_args(parser)


if __name__ == "__main__":
    main()
