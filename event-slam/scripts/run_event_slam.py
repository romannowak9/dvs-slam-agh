#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))


from event_slam.io.result_io import save_outputs
from event_slam.setup import create_pipeline


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    pipeline = None
    interrupted = False

    try:
        pipeline = create_pipeline(config)
        started = time.perf_counter()
        pipeline.run()
        total_seconds = time.perf_counter() - started
        reader_seconds = getattr(pipeline.window_builder, "read_seconds", None)
        if reader_seconds is not None:
            print(f"M3ED reader time: {reader_seconds:.3f} s")
            print(f"Event processing and SLAM time: {total_seconds - reader_seconds:.3f} s")

    except KeyboardInterrupt:
        interrupted = True
        print()
        print("Interrupted by user. Saving partial outputs...")

    finally:
        if pipeline is not None:
            save_outputs(pipeline, config, interrupted)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the offline event-based stereo SLAM pipeline."
    )

    parser.add_argument(
        "--config",
        default=Path("configs/evslam_seq007_test_slam.yaml"),
        type=Path,
        help="Path to the YAML configuration file.",
    )

    return parser.parse_args()


def load_config(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a non-empty YAML dictionary: {path}")
    return config


if __name__ == "__main__":
    main()
