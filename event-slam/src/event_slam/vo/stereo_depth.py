from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import cv2


@dataclass
class StereoDepthStats:
    """
    Basic statistics for sparse stereo matching and triangulation.
    """

    input_count: int = 0
    matched_count: int = 0
    triangulated_count: int = 0

    depth_min: float | None = None
    depth_max: float | None = None
    depth_median: float | None = None

    disparity_min: float | None = None
    disparity_max: float | None = None
    disparity_median: float | None = None


@dataclass
class StereoDepthResult:
    """
    Sparse stereo depth result.

    points_3d_left_camera are expressed in the rectified left camera coordinate frame
    when P1 and P2 come from OpenCV stereoRectify.
    """

    points_2d_left: np.ndarray
    points_2d_right: np.ndarray
    points_3d_left_camera: np.ndarray

    valid_mask: np.ndarray
    epipolar_error: np.ndarray
    disparity: np.ndarray

    stats: StereoDepthStats


class StereoDepthEstimator:
    """
    Sparse stereo matcher and triangulator for rectified stereo event frames.

    Matching is done with pyramidal Lucas-Kanade from left image to right image.
    Since the images are rectified, valid correspondences should have small
    vertical error and a reasonable horizontal disparity.
    """

    def __init__(
        self,
        P1: np.ndarray,
        P2: np.ndarray,
        epipolar_threshold: float = 2.0,
        min_disparity: float = 0.5,
        max_disparity: float | None = 250.0,
        min_depth: float = 0.05,
        max_depth: float | None = 100.0,
        lk_win_size: tuple = (21, 21),
        lk_max_level: int = 3,
    ) -> None:
        
        self.P1 = np.asarray(P1, dtype=np.float64)
        self.P2 = np.asarray(P2, dtype=np.float64)

        if self.P1.shape != (3, 4):
            raise ValueError(f"P1 must have shape (3, 4), got {self.P1.shape}")

        if self.P2.shape != (3, 4):
            raise ValueError(f"P2 must have shape (3, 4), got {self.P2.shape}")

        self.epipolar_threshold = float(epipolar_threshold)
        self.min_disparity = float(min_disparity)
        self.max_disparity = None if max_disparity is None else float(max_disparity)
        self.min_depth = float(min_depth)
        self.max_depth = None if max_depth is None else float(max_depth)

        self.lk_params = {
            "winSize": tuple(lk_win_size),
            "maxLevel": int(lk_max_level),
            "criteria": (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        }

    def estimate(
        self,
        left_img: np.ndarray,
        right_img: np.ndarray,
        left_points: np.ndarray,
    ) -> StereoDepthResult:
        """
        Match left 2D points in the right rectified image and triangulate them.
        """
        left_gray = self._to_gray(left_img)
        right_gray = self._to_gray(right_img)

        left_xy = _as_points_xy(left_points)
        input_count = len(left_xy)

        if input_count == 0:
            return self._empty_result(input_count=0)

        left_lk = left_xy.reshape(-1, 1, 2).astype(np.float32)

        right_lk, status, _ = cv2.calcOpticalFlowPyrLK(
            left_gray,
            right_gray,
            left_lk,
            None,
            **self.lk_params,
        )

        if right_lk is None or status is None:
            return self._empty_result(input_count=input_count)

        right_xy = _as_points_xy(right_lk)

        match_mask = status.reshape(-1).astype(np.bool_)
        match_mask &= self._inside_image(left_xy, left_gray.shape)
        match_mask &= self._inside_image(right_xy, right_gray.shape)

        epipolar_error = np.full(input_count, np.nan, dtype=np.float64)
        disparity = np.full(input_count, np.nan, dtype=np.float64)

        epipolar_error[:] = np.abs(left_xy[:, 1] - right_xy[:, 1])
        disparity[:] = left_xy[:, 0] - right_xy[:, 0]

        match_mask &= epipolar_error <= self.epipolar_threshold
        match_mask &= disparity >= self.min_disparity

        if self.max_disparity is not None:
            match_mask &= disparity <= self.max_disparity

        matched_count = int(np.count_nonzero(match_mask))

        if matched_count == 0:
            return self._empty_result(
                input_count=input_count,
                epipolar_error=epipolar_error,
                disparity=disparity,
            )

        matched_indices = np.where(match_mask)[0]

        points_3d = self._triangulate(
            left_xy[matched_indices],
            right_xy[matched_indices],
        )

        depth_mask = np.isfinite(points_3d).all(axis=1)
        depth_mask &= points_3d[:, 2] > self.min_depth

        if self.max_depth is not None:
            depth_mask &= points_3d[:, 2] <= self.max_depth

        final_indices = matched_indices[depth_mask]

        valid_mask = np.zeros(input_count, dtype=np.bool_)
        valid_mask[final_indices] = True

        final_left = left_xy[final_indices]
        final_right = right_xy[final_indices]
        final_points_3d = points_3d[depth_mask]

        stats = _make_stats(
            input_count=input_count,
            matched_count=matched_count,
            points_3d=final_points_3d,
            disparity=disparity[final_indices],
        )

        return StereoDepthResult(
            points_2d_left=final_left,
            points_2d_right=final_right,
            points_3d_left_camera=final_points_3d,
            valid_mask=valid_mask,
            epipolar_error=epipolar_error,
            disparity=disparity,
            stats=stats,
        )

    def _triangulate(
        self,
        points_left: np.ndarray,
        points_right: np.ndarray,
    ) -> np.ndarray:
        points_left_2xn = points_left.T.astype(np.float64)
        points_right_2xn = points_right.T.astype(np.float64)

        points_4d = cv2.triangulatePoints(
            self.P1,
            self.P2,
            points_left_2xn,
            points_right_2xn,
        )

        w = points_4d[3]
        valid_w = np.abs(w) > 1e-12

        points_3d = np.full((points_4d.shape[1], 3), np.nan, dtype=np.float64)
        points_3d[valid_w] = (points_4d[:3, valid_w] / w[valid_w]).T

        return points_3d

    def _inside_image(self, points: np.ndarray, image_shape: tuple) -> np.ndarray:
        height, width = int(image_shape[0]), int(image_shape[1])

        return (
            (points[:, 0] >= 0.0)
            & (points[:, 0] < width)
            & (points[:, 1] >= 0.0)
            & (points[:, 1] < height)
        )

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image.astype(np.uint8, copy=False)

        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        raise ValueError(f"Unsupported image shape: {image.shape}")

    def _empty_result(
        self,
        input_count: int,
        epipolar_error: np.ndarray | None = None,
        disparity: np.ndarray | None = None,
    ) -> StereoDepthResult:
        if epipolar_error is None:
            epipolar_error = np.full(input_count, np.nan, dtype=np.float64)

        if disparity is None:
            disparity = np.full(input_count, np.nan, dtype=np.float64)

        return StereoDepthResult(
            points_2d_left=_empty_points2(),
            points_2d_right=_empty_points2(),
            points_3d_left_camera=np.empty((0, 3), dtype=np.float64),
            valid_mask=np.zeros(input_count, dtype=np.bool_),
            epipolar_error=epipolar_error,
            disparity=disparity,
            stats=StereoDepthStats(input_count=input_count),
        )


def _as_points_xy(points: np.ndarray) -> np.ndarray:
    if points is None:
        return _empty_points2()

    points = np.asarray(points, dtype=np.float32)

    if points.size == 0:
        return _empty_points2()

    return points.reshape(-1, 2)


def _empty_points2() -> np.ndarray:
    return np.empty((0, 2), dtype=np.float32)


def _make_stats(
    input_count: int,
    matched_count: int,
    points_3d: np.ndarray,
    disparity: np.ndarray,
) -> StereoDepthStats:
    stats = StereoDepthStats(
        input_count=int(input_count),
        matched_count=int(matched_count),
        triangulated_count=int(len(points_3d)),
    )

    if len(points_3d) > 0:
        depth = points_3d[:, 2]

        stats.depth_min = float(np.min(depth))
        stats.depth_max = float(np.max(depth))
        stats.depth_median = float(np.median(depth))

    if len(disparity) > 0:
        stats.disparity_min = float(np.min(disparity))
        stats.disparity_max = float(np.max(disparity))
        stats.disparity_median = float(np.median(disparity))

    return stats
