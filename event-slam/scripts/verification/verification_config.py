from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/evslam_seq007_test_slam.yaml"


def verification_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def load_args(parser: argparse.ArgumentParser) -> tuple:
    """Combine existing project YAML fields with debug-only CLI arguments."""
    cli = vars(parser.parse_args())
    config_path = cli.pop("config")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a non-empty YAML dictionary: {config_path}")

    dataset = config.get("dataset", {})
    imu = config.get("imu", {})
    baf = config.get("baf", {})
    motion_compensation = imu.get("motion_compensation", {})

    params = {}
    for section in (
        "processing",
        "aggregation",
        "rectification",
        "feature_tracker",
        "stereo_depth",
        "pnp",
        "output",
    ):
        params.update(config.get(section, {}))

    params.update(
        bag=Path(dataset["bag_path"]),
        camera_yaml=Path(dataset["camera_yaml"]),
        camera_calibration=Path(dataset["camera_yaml"]),
        imu_yaml=_path_or_none(imu.get("calibration_yaml")),
        imu_calibration=_path_or_none(imu.get("calibration_yaml")),
        left_topic=dataset["left_event_topic"],
        right_topic=dataset["right_event_topic"],
        imu_topic=imu.get("topic"),
        use_baf=bool(baf.get("enabled", False)),
        baf_time_window=baf.get("time_window"),
        baf_radius=baf.get("radius"),
        baf_min_neighbors=baf.get("min_neighbors"),
        reference_time=motion_compensation.get("reference_time", "middle"),
        time_bins=motion_compensation.get("time_bins", 32),
    )
    params.update(cli)
    if params.get("output_dir") is not None:
        params["output_dir"] = Path(params["output_dir"])
    return config, SimpleNamespace(**params)


def _path_or_none(value):
    return None if value is None else Path(value)
