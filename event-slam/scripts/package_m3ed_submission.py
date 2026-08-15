#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))


from event_slam.io.result_io import validate_m3ed_result_file


SEQUENCES = (
    "car_urban_day_ucity_big_loop",
    "falcon_outdoor_day_fast_flight_3",
    "spot_outdoor_day_penno_building_loop",
)


def main() -> None:
    args = parse_args()
    result_paths = []
    for sequence_name in SEQUENCES:
        result_path = args.results_dir / f"{sequence_name}.txt"
        reference_path = args.reference_dir / f"{sequence_name}_ts.txt"
        if not result_path.is_file():
            raise FileNotFoundError(f"Missing challenge result: {result_path}")
        if not reference_path.is_file():
            raise FileNotFoundError(f"Missing reference timestamps: {reference_path}")
        validate_m3ed_result_file(
            result_path,
            sequence_name=sequence_name,
            reference_path=reference_path,
        )
        result_paths.append(result_path)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as archive:
        for result_path in result_paths:
            archive.write(result_path, arcname=result_path.name)
    print(f"M3ED submission saved: {args.output}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate and package the three M3ED SLAM Challenge files."
    )
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("/data/m3ed/ref"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
