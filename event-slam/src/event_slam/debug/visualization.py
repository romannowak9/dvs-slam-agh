from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from event_slam.core.geometry import as_points_xy
from event_slam.events.event_aggregator import BACKGROUND_INTENSITY


def colorize_event_frame(gray: np.ndarray) -> np.ndarray:
    """
    Convert a single-channel event frame to a BGR debug image.

    Convention:
        positive events -> blue,
        negative events -> red,
        background      -> black.
    """
    if gray.ndim != 2:
        raise ValueError(f"Expected a single-channel image, got shape {gray.shape}")

    gray_i16 = gray.astype(np.int16)

    positive = gray_i16 > BACKGROUND_INTENSITY
    negative = gray_i16 < BACKGROUND_INTENSITY

    positive_strength = np.zeros_like(gray, dtype=np.uint8)
    negative_strength = np.zeros_like(gray, dtype=np.uint8)

    positive_strength[positive] = np.clip(
        2 * (gray_i16[positive] - BACKGROUND_INTENSITY),
        0,
        255,
    ).astype(np.uint8)

    negative_strength[negative] = np.clip(
        2 * (BACKGROUND_INTENSITY - gray_i16[negative]),
        0,
        255,
    ).astype(np.uint8)

    color = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    color[:, :, 0] = positive_strength
    color[:, :, 2] = negative_strength

    return color


def make_side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """
    Concatenate two images horizontally.
    """
    if left.shape[0] != right.shape[0]:
        raise ValueError(
            f"Images must have the same height, got {left.shape[0]} and {right.shape[0]}"
        )

    return np.concatenate((left, right), axis=1)


def draw_text_bar(
    image: np.ndarray,
    text: str,
    height: int = 28,
    origin: tuple = (8, 19),
) -> None:
    """
    Draw a black status bar with white text at the top of an image.
    """
    cv2.rectangle(image, (0, 0), (image.shape[1], height), (0, 0, 0), -1)
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_points(
    image: np.ndarray,
    points: np.ndarray,
    color: tuple = (0, 255, 255),
    radius: int = 2,
    thickness: int = -1,
) -> None:
    """
    Draw 2D points on an image.
    """
    for point in as_points_xy(points):
        x, y = int(round(point[0])), int(round(point[1]))
        cv2.circle(image, (x, y), radius, color, thickness, cv2.LINE_AA)


def draw_tracks(
    image: np.ndarray,
    prev_points: np.ndarray,
    curr_points: np.ndarray,
    line_color: tuple = (0, 255, 0),
    point_color: tuple = (0, 255, 255),
    radius: int = 2,
) -> None:
    """
    Draw 2D tracks as lines from previous to current point positions.
    """
    prev_points = as_points_xy(prev_points)
    curr_points = as_points_xy(curr_points)

    count = min(len(prev_points), len(curr_points))

    for index in range(count):
        prev_pt = prev_points[index]
        curr_pt = curr_points[index]

        x0, y0 = int(round(prev_pt[0])), int(round(prev_pt[1]))
        x1, y1 = int(round(curr_pt[0])), int(round(curr_pt[1]))

        cv2.line(image, (x0, y0), (x1, y1), line_color, 1, cv2.LINE_AA)
        cv2.circle(image, (x1, y1), radius, point_color, -1, cv2.LINE_AA)


def draw_stereo_matches(
    image: np.ndarray,
    points_left: np.ndarray,
    points_right: np.ndarray,
    left_width: int,
    points_3d: np.ndarray | None = None,
    max_matches: int | None = None,
) -> None:
    """
    Draw stereo matches on a side-by-side left/right image.
    """
    points_left = as_points_xy(points_left)
    points_right = as_points_xy(points_right)
    count = min(len(points_left), len(points_right))

    if max_matches is not None:
        count = min(count, int(max_matches))

    if count <= 0:
        return

    colors = _make_match_colors(count, points_3d)

    for index in range(count):
        left_pt = points_left[index]
        right_pt = points_right[index]
        x0, y0 = int(round(left_pt[0])), int(round(left_pt[1]))
        x1 = int(round(right_pt[0])) + int(left_width)
        y1 = int(round(right_pt[1]))
        color = colors[index]

        cv2.circle(image, (x0, y0), 2, color, -1, cv2.LINE_AA)
        cv2.circle(image, (x1, y1), 2, color, -1, cv2.LINE_AA)
        cv2.line(image, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)


def show_image(window_name: str, image: np.ndarray, delay_ms: int = 1) -> bool:
    """
    Show an image and return False when user presses q or Esc.
    """
    cv2.imshow(window_name, image)

    key = cv2.waitKey(max(0, int(delay_ms))) & 0xFF

    if key in (27, ord("q")):
        return False

    return True


def save_image(path: Path, image: np.ndarray) -> None:
    """
    Save an image and raise an exception if OpenCV fails.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(str(path), image)

    if not ok:
        raise IOError(f"Could not save image: {path}")


def format_value(value, precision: int = 3) -> str:
    """
    Format a scalar debug value.
    """
    if value is None:
        return "None"

    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return str(value)

    if not np.isfinite(value_float):
        return "nan"

    return f"{value_float:.{precision}f}"


def print_slam_frame(frame_index, window, result, motion_compensation=None) -> None:
    """Print diagnostics produced for one stereo SLAM frame."""
    t = result.T_W_Cleft[:3, 3]
    motion_text = ""

    if motion_compensation is not None:
        left_stats = motion_compensation.left_stats
        right_stats = motion_compensation.right_stats
        motion_text = (
            f"mc_left={left_stats.output_count}/{left_stats.input_count}, "
            f"mc_right={right_stats.output_count}/{right_stats.input_count}, "
        )

    print(
        f"frame {frame_index:05d}: "
        f"timestamp={result.timestamp:.9f}, "
        f"events_left={len(window.left)}, "
        f"events_right={len(window.right)}, "
        f"{motion_text}"
        f"success={result.success}, "
        f"source={result.pose_source}, "
        f"tracks={result.track_count}, "
        f"added={result.new_feature_count}, "
        f"pnp={result.pnp_point_count}, "
        f"inliers={result.pnp_inlier_count}, "
        f"map={result.map_inlier_count}/{result.map_point_count}, "
        f"local={result.local_landmark_count}, "
        f"orb={result.map_descriptor_match_count}, "
        f"loop={result.loop_accepted}, "
        f"relocalized={result.relocalized}, "
        f"map_msg={result.map_message}, "
        f"err_med={format_value(result.reprojection_error_median)}, "
        f"pnp_rot={format_value(result.pnp_rotation_step_deg)}, "
        f"imu_rot={format_value(result.imu_rotation_step_deg)}, "
        f"pnp_imu_err={format_value(result.pnp_imu_rotation_error_deg)}, "
        f"imu_ok={result.imu_rotation_consistent}, "
        f"imu_rejected={result.imu_rejected}, "
        f"depth={result.depth_count}, "
        f"pos=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}], "
        f"msg={result.message}"
    )


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_vector(name: str, vector: np.ndarray) -> None:
    print(f"{name}: {np.array2string(vector, precision=9, suppress_small=False)}")


def _make_match_colors(count: int, points_3d: np.ndarray | None) -> list:
    if points_3d is None or len(points_3d) == 0:
        return [(0, 255, 255)] * count

    points_3d = np.asarray(points_3d, dtype=np.float64)

    if len(points_3d) < count:
        return [(0, 255, 255)] * count

    depths = points_3d[:count, 2]

    if not np.isfinite(depths).any():
        return [(0, 255, 255)] * count

    depth_min = float(np.nanmin(depths))
    depth_max = float(np.nanmax(depths))
    denom = max(1e-9, depth_max - depth_min)
    colors = []

    for depth in depths:
        if not np.isfinite(depth):
            colors.append((0, 255, 255))
            continue

        intensity = int(round(255.0 * (float(depth) - depth_min) / denom))
        intensity = int(np.clip(intensity, 0, 255))
        colors.append((0, 255 - intensity, 255))

    return colors
