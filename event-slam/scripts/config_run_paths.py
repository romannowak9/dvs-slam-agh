#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--dataset", choices=("evslam", "m3ed"), required=True)
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    dataset = config.get("dataset", {})
    output = config.get("output", {})
    output_dir = Path(output.get("output_dir", "outputs/run"))
    dataset_format = dataset.get("format", "evslam_rosbag")

    if args.dataset == "m3ed":
        if dataset_format != "m3ed_h5":
            raise ValueError("M3ED runner requires dataset.format: m3ed_h5")
        sequence_name = dataset.get("sequence_name")
        if not sequence_name:
            raise ValueError("M3ED runner requires dataset.sequence_name")
        raw_result = output_dir / f"{sequence_name}.txt"
        prefix = sequence_name
    else:
        if dataset_format == "m3ed_h5":
            raise ValueError("EvSLAM runner cannot use an M3ED configuration")
        raw_result = output_dir / output.get("result_txt", "result.txt")
        prefix = Path(dataset.get("bag_path", "evslam")).stem

    ground_truth = config.get("evaluation", {}).get("ground_truth_path") or ""
    print(output_dir)
    print(raw_result)
    print(prefix)
    print(ground_truth)


if __name__ == "__main__":
    main()
