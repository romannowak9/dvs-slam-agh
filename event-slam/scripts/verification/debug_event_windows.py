#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"

if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))


from event_slam.debug.visualization import print_section
from event_slam.datasets.evslam_reader import EvSlamRosbagReader
from event_slam.events.event_window import StereoEventWindowBuilder
from verification_config import load_args, verification_parser


def main() -> None:
    _, args = parse_args()

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
        drop_empty_windows=args.drop_empty_windows,
    )

    if args.summary:
        run_summary(builder)
    else:
        inspect_windows(
            builder=builder,
            num_windows=args.num_windows,
            sample_events=args.sample_events,
        )


def inspect_windows(
    builder: StereoEventWindowBuilder,
    num_windows: int,
    sample_events: int,
) -> None:
    print_section("Stereo event windows")

    emitted = 0

    for window_index, window in enumerate(builder.iter_windows()):
        print()
        print(f"window_index: {window_index}")
        print(f"t_start [s]: {window.t_start:.9f}")
        print(f"t_end   [s]: {window.t_end:.9f}")
        print(f"duration [s]: {window.duration:.9f}")
        print(f"left_event_count: {len(window.left)}")
        print(f"right_event_count: {len(window.right)}")

        print_batch_time_range("left", window.left)
        print_batch_time_range("right", window.right)

        print_sample_events("left", window.left, sample_events)
        print_sample_events("right", window.right, sample_events)

        emitted += 1

        if emitted >= num_windows:
            break

    print_stats(builder)


def run_summary(builder: StereoEventWindowBuilder) -> None:
    print_section("Stereo event window summary")

    for _ in builder.iter_windows():
        pass

    print_stats(builder)


def print_stats(builder: StereoEventWindowBuilder) -> None:
    stats = builder.stats

    print()
    print_section("Builder statistics")
    print(f"generated_windows: {stats.generated_windows}")
    print(f"emitted_windows: {stats.emitted_windows}")
    print(f"empty_windows: {stats.empty_windows}")
    print(f"dropped_empty_windows: {stats.dropped_empty_windows}")
    print(f"left_events: {stats.left_events}")
    print(f"right_events: {stats.right_events}")

    if stats.first_window_start is None:
        print("first_window_start [s]: None")
    else:
        print(f"first_window_start [s]: {stats.first_window_start:.9f}")

    if stats.last_window_end is None:
        print("last_window_end [s]: None")
    else:
        print(f"last_window_end [s]: {stats.last_window_end:.9f}")


def print_batch_time_range(name: str, batch) -> None:
    if batch.time_range is None:
        print(f"{name}_time_range [s]: None")
        return

    t_start, t_end = batch.time_range
    print(f"{name}_time_range [s]: {t_start:.9f} -> {t_end:.9f}")


def print_sample_events(name: str, batch, sample_events: int) -> None:
    count = min(len(batch), sample_events)

    if count == 0:
        print(f"{name}_sample_events: []")
        return

    print(f"{name}_sample_events:")

    for index in range(count):
        print(
            f"  [{index}] "
            f"t={batch.t[index]:.9f}, "
            f"x={int(batch.x[index])}, "
            f"y={int(batch.y[index])}, "
            f"p={int(batch.p[index])}"
        )


def parse_args() -> tuple:
    parser = verification_parser(
        "Debug fixed-time stereo event windows from an EvSLAM bag."
    )
    parser.add_argument("--num-windows", default=10, type=int)
    parser.add_argument("--sample-events", default=5, type=int)
    parser.add_argument("--drop-empty-windows", action="store_true")
    parser.add_argument("--summary", action="store_true")
    return load_args(parser)


if __name__ == "__main__":
    main()
