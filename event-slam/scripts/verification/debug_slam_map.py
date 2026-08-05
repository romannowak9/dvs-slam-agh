#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
import mpl_toolkits.mplot3d  # Register the Matplotlib 3D projection.
import numpy as np
import yaml


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"

if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))


from event_slam.debug.visualization import (
    colorize_event_frame,
    draw_points,
    draw_text_bar,
    save_image,
    show_image,
)
from event_slam.vo.setup import create_pipeline


class SlamMapVisualizer:
    """Save each new keyframe with its landmark observations."""

    def __init__(
        self,
        pipeline,
        output_dir: Path,
        display: bool,
        delay_ms: int,
        max_plot_distance: float,
    ):
        self.pipeline = pipeline
        self.output_dir = Path(output_dir)
        self.display = bool(display)
        self.delay_ms = int(delay_ms)
        self.max_plot_distance = float(max_plot_distance)
        self.saved_keyframe_count = 0
        self.stop_requested = False

        self.keyframe_dir = self.output_dir / "keyframes"
        self.keyframe_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, frame_index, window, result, motion_compensation) -> None:
        sparse_map = self.pipeline.vo.map
        if sparse_map is None or len(sparse_map.keyframes) <= self.saved_keyframe_count:
            return

        keyframe = sparse_map.last_keyframe
        left_frame = self.pipeline.aggregator.aggregate_batch(
            window.left,
            window.t_start,
            window.t_end,
        )
        left_rectified = self.pipeline.rectifier.rectify_left(left_frame.image)
        image = self._draw_keyframe(left_rectified, keyframe, sparse_map)

        path = self.keyframe_dir / (
            f"keyframe_{keyframe.id:04d}_frame_{keyframe.frame_index:05d}.png"
        )
        save_image(path, image)
        self.saved_keyframe_count += 1
        print(f"saved keyframe {keyframe.id}: {path}")

        if self.display and not show_image("debug_slam_map", image, self.delay_ms):
            self.stop_requested = True
            self.pipeline.num_frames = len(self.pipeline.vo.results)

    @staticmethod
    def _draw_keyframe(image_gray, keyframe, sparse_map) -> np.ndarray:
        image = colorize_event_frame(image_gray)
        anchor_ids = np.asarray(
            [
                sparse_map.landmarks[int(landmark_id)].anchor_keyframe_id
                for landmark_id in keyframe.landmark_ids
            ]
        )
        new_mask = anchor_ids == keyframe.id

        draw_points(image, keyframe.points_2d[~new_mask], color=(0, 255, 255), radius=3)
        draw_points(image, keyframe.points_2d[new_mask], color=(0, 255, 0), radius=3)
        draw_text_bar(
            image,
            f"KF {keyframe.id} | frame {keyframe.frame_index} | "
            f"landmarks {keyframe.point_count} | "
            f"new {np.count_nonzero(new_mask)} | "
            f"reobserved {np.count_nonzero(~new_mask)}",
        )
        return image

    def save_map_plot(self) -> Path:
        sparse_map = self.pipeline.vo.map
        all_landmarks = list(sparse_map.landmarks.values())
        landmarks = [
            item
            for item in all_landmarks
            if np.linalg.norm(item.position_C_anchor) <= self.max_plot_distance
        ]
        keyframes = sparse_map.keyframes

        positions_W = np.asarray([item.position_W for item in landmarks])
        observations = np.asarray([item.observation_count for item in landmarks])
        camera_positions = np.asarray([item.T_W_C[:3, 3] for item in keyframes])

        figure = plt.figure(figsize=(10, 8))
        axes = figure.add_subplot(111, projection="3d")
        scatter = axes.scatter(
            positions_W[:, 0],
            positions_W[:, 1],
            positions_W[:, 2],
            c=observations,
            cmap="viridis",
            s=4,
            alpha=0.7,
            label="landmarks",
        )
        axes.plot(
            camera_positions[:, 0],
            camera_positions[:, 1],
            camera_positions[:, 2],
            "r.-",
            linewidth=1.5,
            label="keyframes",
        )
        axes.set_xlabel("world X [m]")
        axes.set_ylabel("world Y [m]")
        axes.set_zlabel("world Z [m]")
        axes.set_title(
            f"Sparse map: {len(keyframes)} keyframes, "
            f"{len(landmarks)}/{len(all_landmarks)} landmarks within "
            f"{self.max_plot_distance:g} m"
        )
        axes.legend()
        figure.colorbar(scatter, ax=axes, label="observation count", shrink=0.7)
        figure.tight_layout()

        path = self.output_dir / "map_3d.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        return path


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not config.get("slam", {}).get("enabled", False):
        raise ValueError("The selected config must have slam.enabled=true")

    if args.num_frames is not None:
        config.setdefault("processing", {})["num_frames"] = args.num_frames

    pipeline = create_pipeline(config)
    visualizer = SlamMapVisualizer(
        pipeline=pipeline,
        output_dir=args.output_dir,
        display=args.display,
        delay_ms=args.display_delay_ms,
        max_plot_distance=args.max_plot_distance,
    )
    pipeline.frame_callback = visualizer
    summary = pipeline.run()

    if args.display:
        cv2.destroyAllWindows()

    if not pipeline.vo.map.keyframes:
        print("No keyframes were created.")
        return

    map_path = visualizer.save_map_plot()
    print()
    print(f"processed_frames: {summary.processed_frames}")
    print(f"keyframes: {summary.keyframe_count}")
    print(f"landmarks: {summary.landmark_count}")
    print(f"map_plot: {map_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize SLAM keyframes, landmark observations and sparse map."
    )
    parser.add_argument(
        "--config",
        default=PROJECT_ROOT / "configs/evslam_seq007_test_imu.yaml",
        type=Path,
    )
    parser.add_argument(
        "--num-frames",
        default=120,
        type=int,
        help="Frames to process; use 0 for the whole configured sequence.",
    )
    parser.add_argument(
        "--output-dir",
        default=PROJECT_ROOT / "outputs/debug_slam_map",
        type=Path,
    )
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--display-delay-ms", default=1, type=int)
    parser.add_argument(
        "--max-plot-distance",
        default=5.0,
        type=float,
        help="Hide landmarks farther from their anchor camera on the 3D plot.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
