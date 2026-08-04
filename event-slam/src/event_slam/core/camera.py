from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from event_slam.core.geometry import as_float_array, as_point_array, invert_transform


@dataclass()
class CameraModel:
    """
    A single calibrated camera model.

    The class stores calibration parameters only. Distortion is handled
    by the stereo rectification module.
    """

    name: str
    width: int
    height: int
    K: np.ndarray
    D: np.ndarray
    camera_model: str = "pinhole"
    distortion_model: str = "radtan"

    def __post_init__(self) -> None:
        self.width = int(self.width)
        self.height = int(self.height)

        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"Invalid camera resolution: width={self.width}, height={self.height}"
            )

        self.K = as_float_array(self.K, (3, 3), "K")
        self.D = np.asarray(self.D, dtype=np.float64).reshape(-1)

        if len(self.D) not in (0, 4, 5, 8):
            raise ValueError(
                f"Unexpected distortion vector length: {len(self.D)}. "
                "Expected 0, 4, 5 or 8."
            )

    @classmethod
    def from_intrinsics(
        cls,
        name: str,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        distortion: np.ndarray | None = None,
        camera_model: str = "pinhole",
        distortion_model: str = "radtan",
    ) -> CameraModel:
        K = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        if distortion is None:
            distortion = np.zeros(4, dtype=np.float64)

        return cls(
            name=name,
            width=width,
            height=height,
            K=K,
            D=np.asarray(distortion, dtype=np.float64),
            camera_model=camera_model,
            distortion_model=distortion_model,
        )

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    @property
    def resolution(self) -> tuple[int, int]:
        """
        Return image resolution as (width, height).
        """
        return self.width, self.height

    @property
    def image_shape(self) -> tuple[int, int]:
        """
        Return image shape as (height, width).
        """
        return self.height, self.width

    def copy_with_K(
        self,
        K: np.ndarray,
        width: int | None = None,
        height: int | None = None,
        name: str | None = None,
    ) -> CameraModel:
        """
        Return a copy with a new intrinsic matrix.

        This is useful after rectification, where the effective camera matrix changes.
        """
        return CameraModel(
            name=self.name if name is None else name,
            width=self.width if width is None else width,
            height=self.height if height is None else height,
            K=np.asarray(K, dtype=np.float64),
            D=np.zeros_like(self.D),
            camera_model=self.camera_model,
            distortion_model="none",
        )

    def pixels_to_normalized(self, points_px: np.ndarray) -> np.ndarray:
        """
        Convert pixel coordinates to normalized camera coordinates.

        This method does not remove distortion. It should be used for undistorted
        or rectified images, or only as a simple geometric helper.
        """
        pts = as_point_array(points_px, 2, "points_px")

        x = (pts[:, 0] - self.cx) / self.fx
        y = (pts[:, 1] - self.cy) / self.fy

        return np.column_stack((x, y))

    def normalized_to_pixels(self, points_norm: np.ndarray) -> np.ndarray:
        """
        Convert normalized camera coordinates to pixel coordinates.
        """
        pts = as_point_array(points_norm, 2, "points_norm")

        u = self.fx * pts[:, 0] + self.cx
        v = self.fy * pts[:, 1] + self.cy

        return np.column_stack((u, v))

    def project_points(
        self,
        points_C: np.ndarray,
        eps: float = 1e-9,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Project 3D points from the camera frame to pixels.

        Returns
        -------
        points_px:
            Array of shape (N, 2).
        valid:
            Boolean mask indicating points with positive depth.
        """
        pts = as_point_array(points_C, 3, "points_C")

        z = pts[:, 2]
        valid = z > eps

        z_safe = np.where(np.abs(z) < eps, eps, z)

        x = pts[:, 0] / z_safe
        y = pts[:, 1] / z_safe

        points_px = self.normalized_to_pixels(np.column_stack((x, y)))

        return points_px, valid

    def in_image(self, points_px: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """
        Check whether points are inside the image bounds.
        """
        pts = as_point_array(points_px, 2, "points_px")

        return (
            (pts[:, 0] >= margin)
            & (pts[:, 0] < self.width - margin)
            & (pts[:, 1] >= margin)
            & (pts[:, 1] < self.height - margin)
        )


@dataclass()
class StereoCalibration:
    """
    Calibration of a stereo camera pair.

    Convention:
    - left camera is the reference camera,
    - right camera is the second camera,
    - T_C_right_C_left transforms points from the left camera frame to the right
      camera frame:

        p_C_right = T_C_right_C_left @ p_C_left

    In Kalibr-style camera chains, the cam1 field T_cn_cnm1 is typically the
    transform from the previous camera to the current camera. For cam1, this
    corresponds to T_C1_C0.
    """

    left: CameraModel
    right: CameraModel
    T_C_right_C_left: np.ndarray

    T_C_left_imu: np.ndarray | None = None
    T_C_right_imu: np.ndarray | None = None

    timeshift_left_imu: float = 0.0
    timeshift_right_imu: float = 0.0

    def __post_init__(self) -> None:
        self.T_C_right_C_left = as_float_array(
            self.T_C_right_C_left, (4, 4), "T_C_right_C_left"
        )

        if self.T_C_left_imu is not None:
            self.T_C_left_imu = as_float_array(
                self.T_C_left_imu, (4, 4), "T_C_left_imu"
            )

        if self.T_C_right_imu is not None:
            self.T_C_right_imu = as_float_array(
                self.T_C_right_imu, (4, 4), "T_C_right_imu"
            )

        self.timeshift_left_imu = float(self.timeshift_left_imu)
        self.timeshift_right_imu = float(self.timeshift_right_imu)

    @property
    def T_C_left_C_right(self) -> np.ndarray:
        return invert_transform(self.T_C_right_C_left)

    @property
    def R_C_right_C_left(self) -> np.ndarray:
        return self.T_C_right_C_left[:3, :3]

    @property
    def t_C_right_C_left(self) -> np.ndarray:
        return self.T_C_right_C_left[:3, 3]

    @property
    def baseline(self) -> float:
        """
        Return the stereo baseline length in meters.
        """
        return float(np.linalg.norm(self.t_C_right_C_left))

    def summary(self) -> str:
        return (
            "StereoCalibration("
            f"left={self.left.name}, right={self.right.name}, "
            f"resolution={self.left.width}x{self.left.height}, "
            f"baseline={self.baseline:.6f} m"
            ")"
        )