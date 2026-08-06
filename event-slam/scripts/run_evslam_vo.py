#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
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
        pipeline.run()

    except KeyboardInterrupt:
        interrupted = True
        print()
        print("Interrupted by user. Saving partial outputs...")

    finally:
        if pipeline is not None:
            save_outputs(pipeline, config, interrupted)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the offline EvSLAM ELOPE-like stereo VO pipeline."
    )

    parser.add_argument(
        "--config",
        default=Path("configs/evslam_seq001.yaml"),
        type=Path,
        help="Path to the YAML configuration file.",
    )

    return parser.parse_args()


def load_config(path: Path) -> dict:
    path = Path(path)

    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Empty config file: {path}")

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML dictionary: {path}")

    return config


if __name__ == "__main__":
    main()
