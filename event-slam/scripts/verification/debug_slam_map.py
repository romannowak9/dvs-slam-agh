#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

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
from event_slam.debug.slam_plots import (
    save_sparse_map_plot,
    save_tracking_diagnostics_plot,
)
from event_slam.io.result_io import save_outputs
from event_slam.setup import create_pipeline
from verification_config import load_args, verification_parser


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
        sparse_map = self.pipeline.slam.map
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
            self.pipeline.num_frames = len(self.pipeline.slam.results)

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
        sparse_map = self.pipeline.slam.map
        landmarks = list(sparse_map.landmarks.values())
        return save_sparse_map_plot(
            keyframe_positions_W=np.asarray(
                [item.T_W_C[:3, 3] for item in sparse_map.keyframes]
            ),
            landmark_positions_W=np.asarray(
                [item.position_W for item in landmarks]
            ),
            landmark_positions_C_anchor=np.asarray(
                [item.position_C_anchor for item in landmarks]
            ),
            observation_counts=np.asarray(
                [item.observation_count for item in landmarks]
            ),
            path=self.output_dir / "map_3d.png",
            max_plot_distance=self.max_plot_distance,
        )

    def save_tracking_plot(self) -> Path:
        """Plot map correspondences and the selected pose source per frame."""
        results = self.pipeline.slam.results
        return save_tracking_diagnostics_plot(
            track_counts=np.asarray([item.track_count for item in results]),
            new_feature_counts=np.asarray(
                [item.new_feature_count for item in results]
            ),
            map_point_counts=np.asarray([item.map_point_count for item in results]),
            map_inlier_counts=np.asarray(
                [item.map_inlier_count for item in results]
            ),
            pnp_inlier_counts=np.asarray([item.pnp_inlier_count for item in results]),
            pose_sources=np.asarray([item.pose_source for item in results]),
            descriptor_match_counts=np.asarray(
                [item.map_descriptor_match_count for item in results]
            ),
            path=self.output_dir / "tracking_diagnostics.png",
        )


def main() -> None:
    config, args = parse_args()

    if not config.get("slam", {}).get("enabled", False):
        raise ValueError("The selected config must have slam.enabled=true")

    if args.num_frames is not None:
        config.setdefault("processing", {})["num_frames"] = args.num_frames
    config.setdefault("output", {})["output_dir"] = str(args.output_dir)

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

    if not pipeline.slam.map.keyframes:
        print("No keyframes were created.")
        return

    map_path = visualizer.save_map_plot()
    tracking_path = visualizer.save_tracking_plot()
    save_outputs(pipeline, config)
    with (args.output_dir / "run_config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)
    print()
    print(f"processed_frames: {summary.processed_frames}")
    print(f"keyframes: {summary.keyframe_count}")
    print(f"landmarks: {summary.landmark_count}")
    print(f"map_plot: {map_path}")
    print(f"tracking_plot: {tracking_path}")


def parse_args() -> tuple:
    parser = verification_parser(
        "Visualize SLAM keyframes, observations and sparse map."
    )
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--display-delay-ms", default=1, type=int)
    parser.add_argument("--max-plot-distance", default=5.0, type=float)
    return load_args(parser)


if __name__ == "__main__":
    main()
