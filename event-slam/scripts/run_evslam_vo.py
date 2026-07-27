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


from event_slam.io.result_writer import write_result_from_reference_file
from event_slam.vo.pipeline import EvSlamStereoVOPipeline


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    pipeline = None
    interrupted = False

    try:
        pipeline = EvSlamStereoVOPipeline(config)
        pipeline.run()

    except KeyboardInterrupt:
        interrupted = True
        print()
        print("Interrupted by user. Saving partial outputs...")

    finally:
        if pipeline is not None:
            csv_path, velocity_path, result_path, result_stats = save_outputs(
                pipeline=pipeline,
                config=config,
            )

            pipeline.print_summary()

            print()
            print("Saved outputs")
            print("=" * 80)
            print(f"trajectory_csv: {csv_path}")

            if velocity_path is not None:
                print(f"velocity_csv: {velocity_path}")

            if result_path is not None:
                print(f"result_txt: {result_path}")

                if result_stats is not None:
                    print(
                        "result_stats: "
                        f"reference={result_stats.reference_count}, "
                        f"written={result_stats.written_count}, "
                        f"skipped={result_stats.skipped_count}"
                    )

            if interrupted:
                print()
                print("Partial outputs saved after KeyboardInterrupt.")


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


def save_outputs(
    pipeline: EvSlamStereoVOPipeline,
    config: dict,
) -> tuple:
    csv_path = pipeline.save_trajectory_output()
    velocity_path = pipeline.save_velocity_output()
    result_path, result_stats = save_challenge_result(
        pipeline=pipeline,
        config=config,
    )

    return csv_path, velocity_path, result_path, result_stats


def save_challenge_result(
    pipeline: EvSlamStereoVOPipeline,
    config: dict,
) -> tuple:
    dataset_cfg = config.get("dataset", {})
    reference_path = dataset_cfg.get("reference_timestamps_path")

    if not reference_path:
        return None, None

    if len(pipeline.trajectory) == 0:
        print("Skipping challenge result: trajectory is empty.")
        return None, None

    result_path = pipeline.get_output_path("result_txt", "result.txt")

    result_stats = write_result_from_reference_file(
        trajectory=pipeline.trajectory,
        velocity=pipeline.velocity_trajectory,
        reference_path=reference_path,
        output_path=result_path,
        skip_out_of_range=True,
    )

    return result_path, result_stats


if __name__ == "__main__":
    main()