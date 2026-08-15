#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import resource
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))


from event_slam.calibration.m3ed_parser import (
    load_m3ed_imu_calibration,
    load_m3ed_stereo_calibration,
)
from event_slam.datasets.m3ed_reader import M3edH5Reader


DEFAULT_H5 = Path("/data/m3ed/falcon_outdoor_day_fast_flight_2_data.h5")
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/m3ed_reader_benchmark/benchmark.txt"
RAW_EVENT_BYTES = 13


def main() -> None:
    args = parse_args()
    report = benchmark(args.path, args.time_window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nsaved: {args.output}")


def benchmark(path: Path, time_window: float) -> str:
    full_reader = M3edH5Reader(path, time_window, drop_empty_windows=False)
    common_start, common_end = full_reader.get_time_range()
    middle = 0.5 * (common_start + common_end) - 0.5 * time_window
    starts = (
        ("first", common_start),
        ("middle", middle),
        ("last", common_end - time_window),
    )

    lines = [
        "M3ED H5 reader benchmark",
        f"path: {path}",
        f"common_range_s: {common_start:.6f} -> {common_end:.6f}",
        f"time_window_s: {time_window:.6f}",
    ]
    for label, start in starts:
        gc.collect()
        reader = M3edH5Reader(
            path,
            time_window,
            t_start=start,
            t_end=start + time_window,
            drop_empty_windows=False,
        )
        before_rss = current_rss_mib()
        started = time.perf_counter()
        window = next(reader.iter_windows())
        elapsed = time.perf_counter() - started
        validate_window(window)

        event_count = len(window.left) + len(window.right)
        payload_mib = event_count * RAW_EVENT_BYTES / (1024.0**2)
        throughput = payload_mib / elapsed if elapsed > 0.0 else float("inf")
        lines.extend(
            (
                "",
                f"[{label}]",
                f"range_s: {window.t_start:.6f} -> {window.t_end:.6f}",
                f"left_events: {len(window.left)}",
                f"right_events: {len(window.right)}",
                f"total_events: {event_count}",
                f"raw_payload_mib: {payload_mib:.3f}",
                f"read_s: {elapsed:.6f}",
                f"throughput_mib_s: {throughput:.3f}",
                f"rss_before_mib: {before_rss:.3f}",
                f"rss_after_mib: {current_rss_mib():.3f}",
            )
        )
        del window, reader

    lines.extend(("", "half_open_window_boundaries: passed"))
    stereo = load_m3ed_stereo_calibration(path)
    imu = load_m3ed_imu_calibration(path)
    imu_t, imu_omega = full_reader.load_imu_gyro()
    imu_rate = 1.0 / np.median(np.diff(imu_t))
    lines.extend(
        (
            "",
            "[calibration_and_imu]",
            f"resolution: {stereo.left.width}x{stereo.left.height}",
            f"baseline_m: {stereo.baseline:.9f}",
            f"T_Pright_Pleft_tx_m: {stereo.T_C_right_C_left[0, 3]:.9f}",
            f"imu_topic: {imu.topic}",
            f"imu_samples: {len(imu_t)}",
            f"imu_range_s: {imu_t[0]:.6f} -> {imu_t[-1]:.6f}",
            f"imu_rate_hz: {imu_rate:.3f}",
            f"imu_omega_shape: {imu_omega.shape}",
            f"peak_rss_mib: {peak_rss_mib():.3f}",
        )
    )
    return "\n".join(lines)


def validate_window(window) -> None:
    for batch in (window.left, window.right):
        if batch.is_empty:
            continue
        if batch.t[0] < window.t_start or batch.t[-1] >= window.t_end:
            raise RuntimeError(
                f"{batch.camera.value} events violate the half-open window boundary"
            )


def current_rss_mib() -> float:
    with Path("/proc/self/statm").open("r", encoding="utf-8") as file:
        resident_pages = int(file.read().split()[1])
    return resident_pages * resource.getpagesize() / (1024.0**2)


def peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect and benchmark three short windows from an M3ED H5 file."
    )
    parser.add_argument("--path", type=Path, default=DEFAULT_H5)
    parser.add_argument("--time-window", type=float, default=0.012)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    main()
