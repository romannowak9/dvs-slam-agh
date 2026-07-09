from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class FeatureDetectorMode(str, Enum):
    FAST = "fast"
    GFTT = "gftt"


@dataclass
class FeatureTrackingResult:
    """
    Result of tracking features from the previous frame to the current frame.
    """

    prev_points: np.ndarray
    curr_points: np.ndarray
    status_mask: np.ndarray

    track_count: int
    active_points: np.ndarray
    active_count: int

    redetected: bool
    detected_count: int


class FeatureTracker:
    """
    Stateful FAST/GFTT + pyramidal Lucas-Kanade feature tracker.

    The logic is intentionally close to the ELOPE-style frontend:
        1. detect features in the first frame,
        2. track them with calcOpticalFlowPyrLK,
        3. reject invalid tracks,
        4. redetect features when too few tracks remain.
    """

    def __init__(
        self,
        detector: FeatureDetectorMode | str = FeatureDetectorMode.FAST,
        min_features: int = 250,
        max_features: int = 1000,
        fast_threshold: int = 25,
        gftt_quality_level: float = 0.01,
        gftt_min_distance: float = 7.0,
        lk_win_size: tuple = (21, 21),
        lk_max_level: int = 3,
        use_forward_backward_check: bool = False,
        fb_threshold: float = 1.0,
    ) -> None:
        self.cv2 = _import_cv2()

        self.detector = FeatureDetectorMode(detector)
        self.min_features = int(min_features)
        self.max_features = int(max_features)
        self.fast_threshold = int(fast_threshold)

        self.gftt_quality_level = float(gftt_quality_level)
        self.gftt_min_distance = float(gftt_min_distance)

        self.lk_params = {
            "winSize": tuple(lk_win_size),
            "maxLevel": int(lk_max_level),
            "criteria": (
                self.cv2.TERM_CRITERIA_EPS | self.cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        }

        self.use_forward_backward_check = bool(use_forward_backward_check)
        self.fb_threshold = float(fb_threshold)

        self.prev_gray = None
        self.prev_points = None

    def reset(self) -> None:
        """
        Clear the internal tracker state.
        """
        self.prev_gray = None
        self.prev_points = None

    def process(self, image: np.ndarray) -> FeatureTrackingResult:
        """
        Process one frame and return tracks from previous frame to current frame.

        On the first call, no temporal tracks exist yet. The tracker only detects
        initial features and stores them for the next frame.
        """
        gray = self._to_gray(image)

        if self.prev_gray is None or self.prev_points is None or len(self.prev_points) == 0:
            detected = self.detect_features(gray)
            self.prev_gray = gray
            self.prev_points = detected

            return _empty_result(
                active_points=_points_to_xy(detected),
                redetected=True,
                detected_count=len(detected),
            )

        prev_points = self.prev_points

        curr_points, status, _ = self.cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            prev_points,
            None,
            **self.lk_params,
        )

        if curr_points is None or status is None:
            tracked_prev = _empty_points()
            tracked_curr = _empty_points()
            status_mask = np.zeros(len(prev_points), dtype=np.bool_)
        else:
            status_mask = status.reshape(-1).astype(np.bool_)
            status_mask &= self._inside_image(curr_points, gray.shape)

            if self.use_forward_backward_check:
                status_mask &= self._forward_backward_mask(
                    prev_gray=self.prev_gray,
                    curr_gray=gray,
                    prev_points=prev_points,
                    curr_points=curr_points,
                )

            tracked_prev = _points_to_xy(prev_points[status_mask])
            tracked_curr = _points_to_xy(curr_points[status_mask])

        redetected = False
        detected_count = 0

        if len(tracked_curr) < self.min_features:
            new_points = self.detect_features(gray)
            self.prev_points = new_points
            redetected = True
            detected_count = len(new_points)
        else:
            self.prev_points = tracked_curr.reshape(-1, 1, 2).astype(np.float32)

        self.prev_gray = gray

        return FeatureTrackingResult(
            prev_points=tracked_prev,
            curr_points=tracked_curr,
            status_mask=status_mask,
            track_count=len(tracked_curr),
            active_points=_points_to_xy(self.prev_points),
            active_count=len(self.prev_points),
            redetected=redetected,
            detected_count=detected_count,
        )

    def detect_features(self, gray: np.ndarray) -> np.ndarray:
        """
        Detect feature points in a grayscale image.
        """
        if self.detector == FeatureDetectorMode.FAST:
            return self._detect_fast(gray)

        if self.detector == FeatureDetectorMode.GFTT:
            return self._detect_gftt(gray)

        raise ValueError(f"Unsupported detector: {self.detector}")

    def _detect_fast(self, gray: np.ndarray) -> np.ndarray:
        detector = self.cv2.FastFeatureDetector_create(
            threshold=self.fast_threshold,
            nonmaxSuppression=True,
        )

        keypoints = detector.detect(gray, None)

        if len(keypoints) == 0:
            return _empty_lk_points()

        keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)
        keypoints = keypoints[: self.max_features]

        points = self.cv2.KeyPoint_convert(keypoints)

        return points.reshape(-1, 1, 2).astype(np.float32)

    def _detect_gftt(self, gray: np.ndarray) -> np.ndarray:
        points = self.cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_features,
            qualityLevel=self.gftt_quality_level,
            minDistance=self.gftt_min_distance,
            blockSize=7,
        )

        if points is None:
            return _empty_lk_points()

        return points.astype(np.float32)

    def _forward_backward_mask(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        prev_points: np.ndarray,
        curr_points: np.ndarray,
    ) -> np.ndarray:
        back_points, back_status, _ = self.cv2.calcOpticalFlowPyrLK(
            curr_gray,
            prev_gray,
            curr_points,
            None,
            **self.lk_params,
        )

        if back_points is None or back_status is None:
            return np.zeros(len(prev_points), dtype=np.bool_)

        prev_xy = _points_to_xy(prev_points)
        back_xy = _points_to_xy(back_points)

        fb_error = np.linalg.norm(prev_xy - back_xy, axis=1)
        fb_ok = fb_error <= self.fb_threshold

        return back_status.reshape(-1).astype(np.bool_) & fb_ok

    def _inside_image(self, points: np.ndarray, image_shape: tuple) -> np.ndarray:
        height, width = int(image_shape[0]), int(image_shape[1])

        xy = _points_to_xy(points)

        return (
            (xy[:, 0] >= 0.0)
            & (xy[:, 0] < width)
            & (xy[:, 1] >= 0.0)
            & (xy[:, 1] < height)
        )

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image.astype(np.uint8, copy=False)

        if image.ndim == 3 and image.shape[2] == 3:
            return self.cv2.cvtColor(image, self.cv2.COLOR_BGR2GRAY)

        raise ValueError(f"Unsupported image shape: {image.shape}")


def _points_to_xy(points: np.ndarray) -> np.ndarray:
    if points is None or len(points) == 0:
        return _empty_points()

    return points.reshape(-1, 2).astype(np.float32)


def _empty_lk_points() -> np.ndarray:
    return np.empty((0, 1, 2), dtype=np.float32)


def _empty_points() -> np.ndarray:
    return np.empty((0, 2), dtype=np.float32)


def _empty_result(
    active_points: np.ndarray,
    redetected: bool,
    detected_count: int,
) -> FeatureTrackingResult:
    return FeatureTrackingResult(
        prev_points=_empty_points(),
        curr_points=_empty_points(),
        status_mask=np.empty(0, dtype=np.bool_),
        track_count=0,
        active_points=active_points,
        active_count=len(active_points),
        redetected=redetected,
        detected_count=detected_count,
    )


def _import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV Python is required. Install it with: apt install python3-opencv"
        ) from exc

    return cv2