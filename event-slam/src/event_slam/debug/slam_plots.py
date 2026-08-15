from __future__ import annotations

from pathlib import Path

import matplotlib
import mpl_toolkits.mplot3d  # Register the Matplotlib 3D projection.
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt


POSE_SOURCES = (
    "none",
    "vo_fallback",
    "map",
    "relocalization",
)


def save_sparse_map_plot(
    keyframe_positions_W: np.ndarray,
    landmark_positions_W: np.ndarray,
    landmark_positions_C_anchor: np.ndarray,
    observation_counts: np.ndarray,
    path: Path,
    max_plot_distance: float = 5.0,
) -> Path:
    """Plot keyframe trajectory and landmarks close to their anchor camera."""
    keyframe_positions_W = _points3(keyframe_positions_W)
    landmark_positions_W = _points3(landmark_positions_W)
    landmark_positions_C_anchor = _points3(landmark_positions_C_anchor)
    observation_counts = np.asarray(observation_counts)

    visible = (
        np.linalg.norm(landmark_positions_C_anchor, axis=1) <= max_plot_distance
    )
    positions_W = landmark_positions_W[visible]
    observations = observation_counts[visible]

    figure = plt.figure(figsize=(10, 8))
    axes = figure.add_subplot(111, projection="3d")
    scatter = axes.scatter(
        positions_W[:, 0],
        positions_W[:, 2],
        positions_W[:, 1],
        c=observations,
        cmap="viridis",
        s=4,
        alpha=0.7,
        label="landmarks",
    )
    axes.plot(
        keyframe_positions_W[:, 0],
        keyframe_positions_W[:, 2],
        keyframe_positions_W[:, 1],
        "r.-",
        linewidth=1.5,
        label="keyframes",
    )
    axes.set_xlabel("world X - right [m]")
    axes.set_ylabel("world Z - forward [m]")
    axes.set_zlabel("world Y - down [m]")
    axes.invert_zaxis()
    axes.set_title(
        f"Sparse map: {len(keyframe_positions_W)} keyframes, "
        f"{len(positions_W)}/{len(landmark_positions_W)} landmarks within "
        f"{max_plot_distance:g} m"
    )
    axes.legend()
    figure.colorbar(scatter, ax=axes, label="observation count", shrink=0.7)
    figure.tight_layout()

    return _save_figure(figure, path)


def save_tracking_diagnostics_plot(
    track_counts: np.ndarray,
    new_feature_counts: np.ndarray,
    map_point_counts: np.ndarray,
    map_inlier_counts: np.ndarray,
    pnp_inlier_counts: np.ndarray,
    pose_sources: np.ndarray,
    descriptor_match_counts: np.ndarray,
    loop_accepted: np.ndarray,
    path: Path,
) -> Path:
    """Plot map correspondences and the selected pose source per frame."""
    frames = np.arange(len(track_counts))
    source_levels = {name: index for index, name in enumerate(POSE_SOURCES)}
    source_values = np.asarray(
        [source_levels.get(name, np.nan) for name in pose_sources]
    )

    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(frames, track_counts, label="tracked features")
    axes[0].plot(
        frames,
        np.asarray(track_counts) + np.asarray(new_feature_counts),
        label="active features",
    )
    axes[0].plot(frames, map_point_counts, label="map points")
    axes[0].plot(frames, map_inlier_counts, label="map inliers")
    axes[0].plot(frames, pnp_inlier_counts, label="used inliers")
    axes[0].set_ylabel("count")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].scatter(
        frames,
        source_values,
        c=source_values,
        cmap="viridis",
        s=18,
    )
    descriptor_frames = np.flatnonzero(np.asarray(descriptor_match_counts) > 0)
    axes[1].scatter(
        descriptor_frames,
        source_values[descriptor_frames],
        marker="x",
        s=60,
        color="red",
        label="ORB recovery",
    )
    loop_frames = np.flatnonzero(np.asarray(loop_accepted, dtype=bool))
    axes[1].scatter(
        loop_frames,
        source_values[loop_frames],
        marker="*",
        s=180,
        color="gold",
        edgecolor="black",
        label="loop closure",
        zorder=3,
    )
    axes[1].set_yticks(list(source_levels.values()))
    axes[1].set_yticklabels(list(source_levels))
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("pose source")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()

    return _save_figure(figure, path)


def _points3(points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64).reshape(-1, 3)


def _save_figure(figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path
