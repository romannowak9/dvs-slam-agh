from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from event_slam.core.camera import StereoCalibration


@dataclass
class StereoRectificationResult:
    """
    Rectification parameters returned by OpenCV.

    P1 and P2 are projection matrices for the rectified left/right cameras.
    Q is the disparity-to-depth reprojection matrix.
    """

    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray

    K_left_rectified: np.ndarray
    K_right_rectified: np.ndarray

    left_map1: np.ndarray
    left_map2: np.ndarray
    right_map1: np.ndarray
    right_map2: np.ndarray

    valid_roi_left: tuple
    valid_roi_right: tuple
    baseline: float


class StereoRectifier:
    """
    OpenCV-based stereo rectifier for the calibrated EvSLAM stereo event cameras.

    The calibration transform T_C_right_C_left is interpreted as:
        X_right = R_C_right_C_left * X_left + t_C_right_C_left
    which matches OpenCV's stereoRectify convention: R and T from camera 1 to camera 2.
    """

    def __init__(
        self,
        calibration: StereoCalibration,
        image_shape: tuple | None = None,
        alpha: float = 0.0,
        interpolation: str = "nearest",
    ) -> None:
        self.cv2 = _import_cv2()
        self.calibration = calibration

        if image_shape is None:
            image_shape = calibration.left.image_shape

        self.height = int(image_shape[0])
        self.width = int(image_shape[1])

        if self.height <= 0 or self.width <= 0:
            raise ValueError(f"Invalid image_shape: {image_shape}")

        self.alpha = float(alpha)
        self.interpolation = _parse_interpolation(self.cv2, interpolation)

        self.result = self._build_rectification()

    @property
    def R1(self) -> np.ndarray:
        return self.result.R1

    @property
    def R2(self) -> np.ndarray:
        return self.result.R2

    @property
    def P1(self) -> np.ndarray:
        return self.result.P1

    @property
    def P2(self) -> np.ndarray:
        return self.result.P2

    @property
    def Q(self) -> np.ndarray:
        return self.result.Q

    @property
    def K_left_rectified(self) -> np.ndarray:
        return self.result.K_left_rectified

    @property
    def K_right_rectified(self) -> np.ndarray:
        return self.result.K_right_rectified

    @property
    def baseline(self) -> float:
        return self.result.baseline

    def rectify_left(self, image: np.ndarray) -> np.ndarray:
        """
        Rectify one left-camera image.
        """
        self._check_image_size(image)

        return self.cv2.remap(
            image,
            self.result.left_map1,
            self.result.left_map2,
            self.interpolation,
        )

    def rectify_right(self, image: np.ndarray) -> np.ndarray:
        """
        Rectify one right-camera image.
        """
        self._check_image_size(image)

        return self.cv2.remap(
            image,
            self.result.right_map1,
            self.result.right_map2,
            self.interpolation,
        )

    def rectify_pair(
        self,
        left_img: np.ndarray,
        right_img: np.ndarray,
    ) -> tuple:
        """
        Rectify a synchronized left/right image pair.
        """
        return self.rectify_left(left_img), self.rectify_right(right_img)

    def _build_rectification(self) -> StereoRectificationResult:
        K1 = self.calibration.left.K.astype(np.float64)
        D1 = self.calibration.left.D.astype(np.float64).reshape(-1, 1)

        K2 = self.calibration.right.K.astype(np.float64)
        D2 = self.calibration.right.D.astype(np.float64).reshape(-1, 1)

        R = self.calibration.T_C_right_C_left[:3, :3].astype(np.float64)
        T = self.calibration.T_C_right_C_left[:3, 3].astype(np.float64).reshape(3, 1)

        image_size = (self.width, self.height)

        R1, R2, P1, P2, Q, roi1, roi2 = self.cv2.stereoRectify(
            cameraMatrix1=K1,
            distCoeffs1=D1,
            cameraMatrix2=K2,
            distCoeffs2=D2,
            imageSize=image_size,
            R=R,
            T=T,
            flags=self.cv2.CALIB_ZERO_DISPARITY,
            alpha=self.alpha,
            newImageSize=image_size,
        )

        K_left_rectified = P1[:3, :3].copy()
        K_right_rectified = P2[:3, :3].copy()

        left_map1, left_map2 = self.cv2.initUndistortRectifyMap(
            cameraMatrix=K1,
            distCoeffs=D1,
            R=R1,
            newCameraMatrix=K_left_rectified,
            size=image_size,
            m1type=self.cv2.CV_16SC2,
        )

        right_map1, right_map2 = self.cv2.initUndistortRectifyMap(
            cameraMatrix=K2,
            distCoeffs=D2,
            R=R2,
            newCameraMatrix=K_right_rectified,
            size=image_size,
            m1type=self.cv2.CV_16SC2,
        )

        baseline = _baseline_from_projection(P2)

        return StereoRectificationResult(
            R1=R1,
            R2=R2,
            P1=P1,
            P2=P2,
            Q=Q,
            K_left_rectified=K_left_rectified,
            K_right_rectified=K_right_rectified,
            left_map1=left_map1,
            left_map2=left_map2,
            right_map1=right_map1,
            right_map2=right_map2,
            valid_roi_left=tuple(roi1),
            valid_roi_right=tuple(roi2),
            baseline=baseline,
        )

    def _check_image_size(self, image: np.ndarray) -> None:
        if image.shape[0] != self.height or image.shape[1] != self.width:
            raise ValueError(
                f"Expected image size {(self.height, self.width)}, "
                f"got {image.shape[:2]}"
            )


def _baseline_from_projection(P2: np.ndarray) -> float:
    tx = 0.0
    ty = 0.0

    if abs(P2[0, 0]) > 1e-12:
        tx = P2[0, 3] / P2[0, 0]

    if abs(P2[1, 1]) > 1e-12:
        ty = P2[1, 3] / P2[1, 1]

    return float(np.sqrt(tx * tx + ty * ty))


def _parse_interpolation(cv2, interpolation: str) -> int:
    if interpolation == "nearest":
        return cv2.INTER_NEAREST

    if interpolation == "linear":
        return cv2.INTER_LINEAR

    raise ValueError(f"Unsupported interpolation: {interpolation}")


def _import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV Python is required. Install it with: apt install python3-opencv"
        ) from exc

    return cv2