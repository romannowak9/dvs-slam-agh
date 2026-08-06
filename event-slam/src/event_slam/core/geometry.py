from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def as_float_array(array: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    if shape == (3,) and all(hasattr(array, attr) for attr in ("x", "y", "z")):
        arr = np.array([array.x, array.y, array.z], dtype=np.float64)
    else:
        arr = np.asarray(array, dtype=np.float64)

    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")

    return arr


def empty_points(dimensions: int, dtype=np.float32) -> np.ndarray:
    """Return an empty point array with the requested dimensionality."""
    return np.empty((0, int(dimensions)), dtype=dtype)


def as_point_array(points: np.ndarray, dimensions: int, name: str) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != dimensions:
        raise ValueError(
            f"{name} must have shape (N, {dimensions}), got {points.shape}"
        )

    return points


def as_points_xy(points: np.ndarray) -> np.ndarray:
    if points is None:
        return empty_points(2)

    points = np.asarray(points, dtype=np.float32)

    if points.size == 0:
        return empty_points(2)

    return points.reshape(-1, 2)


def skew(v: np.ndarray) -> np.ndarray:
    """
    Return the skew-symmetric matrix [v]_x.

    Used for cross products, epipolar geometry, Lie algebra operations
    and Jacobians.
    """
    v = as_float_array(v, (3,), "v")

    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float64,
    )


def rotvec_to_rotmat(rotation_vector: np.ndarray) -> np.ndarray:
    """
    Convert an axis-angle rotation vector to a rotation matrix.

    The input is a rotation vector:
        rotation_vector = axis * angle

    where the vector direction is the rotation axis and its norm is the
    rotation angle in radians.
    """
    rotation_vector = as_float_array(rotation_vector, (3,), "rotation_vector")
    angle = float(np.linalg.norm(rotation_vector))

    if angle < 1e-12:
        return np.eye(3, dtype=np.float64) + skew(rotation_vector)

    axis = rotation_vector / angle
    K = skew(axis)

    return (
        np.eye(3, dtype=np.float64)
        + np.sin(angle) * K
        + (1.0 - np.cos(angle)) * (K @ K)
    )


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Build an SE(3) homogeneous transform:

        T = [R t]
            [0 1]
    """
    R = as_float_array(R, (3, 3), "R")
    t = as_float_array(t, (3,), "t")

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def orthonormalize_rotation(R: np.ndarray) -> np.ndarray:
    """Project a nearly rotational matrix onto SO(3)."""
    U, _, Vt = np.linalg.svd(as_float_array(R, (3, 3), "R"))
    R_normalized = U @ Vt
    if np.linalg.det(R_normalized) < 0.0:
        U[:, -1] *= -1.0
        R_normalized = U @ Vt
    return R_normalized


def invert_transform(T: np.ndarray) -> np.ndarray:
    """
    Invert an SE(3) homogeneous transform.
    """
    T = as_float_array(T, (4, 4), "T")

    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t

    return T_inv


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Transform 3D points using a homogeneous transform.

    Parameters:
    T:
        A 4x4 homogeneous transform.
    points:
        An array of shape (N, 3) or a single point of shape (3,).

    Returns:
    np.ndarray
        Transformed points with the same point layout as the input.
    """
    T = as_float_array(T, (4, 4), "T")
    pts = np.asarray(points, dtype=np.float64)

    single_point = pts.ndim == 1

    if single_point:
        pts = pts.reshape(1, 3)

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3) or (3,), got {points.shape}")

    transformed = (T[:3, :3] @ pts.T).T + T[:3, 3]

    if single_point:
        return transformed.reshape(3)

    return transformed


def normalize_homogeneous(points_h: np.ndarray) -> np.ndarray:
    """
    Convert homogeneous points to Euclidean points.

    Input shape:
        (N, D)

    Output shape:
        (N, D - 1)
    """
    pts = np.asarray(points_h, dtype=np.float64)

    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError(f"points_h must have shape (N, D), got {pts.shape}")

    denominator = pts[:, -1:]
    denominator_safe = np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)

    return pts[:, :-1] / denominator_safe


def quat_xyzw_normalize(q: np.ndarray) -> np.ndarray:
    """
    Normalize a quaternion stored as [qx, qy, qz, qw].
    """
    q = as_float_array(q, (4,), "q")

    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("Cannot normalize a zero quaternion")

    return q / norm


def quat_xyzw_to_rotmat(q: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion [qx, qy, qz, qw] to a 3x3 rotation matrix.
    """
    qx, qy, qz, qw = quat_xyzw_normalize(q)

    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to a quaternion [qx, qy, qz, qw].
    """
    R = as_float_array(R, (3, 3), "R")

    trace = np.trace(R)

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    return quat_xyzw_normalize(np.array([qx, qy, qz, qw], dtype=np.float64))


def slerp_quat_xyzw(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """
    Spherical linear interpolation of two quaternions [qx, qy, qz, qw].
    """
    q0 = quat_xyzw_normalize(q0)
    q1 = quat_xyzw_normalize(q1)

    alpha = float(alpha)
    if alpha <= 0.0:
        return q0
    if alpha >= 1.0:
        return q1

    dot = float(np.dot(q0, q1))

    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        return quat_xyzw_normalize(q0 + alpha * (q1 - q0))

    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)

    theta = theta_0 * alpha
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return quat_xyzw_normalize((s0 * q0) + (s1 * q1))


@dataclass()
class Pose:
    """
    Camera pose represented as T_W_C.

    R:
        R_W_C, rotation from the camera frame C to the world frame W.
    t:
        t_W_C, position of the camera center in the world frame W.
    """

    R: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def __post_init__(self) -> None:
        self.R = as_float_array(self.R, (3, 3), "R")
        self.t = as_float_array(self.t, (3,), "t")

    @classmethod
    def identity(cls) -> Pose:
        return cls()

    @classmethod
    def from_matrix(cls, T_W_C: np.ndarray) -> Pose:
        T_W_C = as_float_array(T_W_C, (4, 4), "T_W_C")
        return cls(R=T_W_C[:3, :3], t=T_W_C[:3, 3])

    @classmethod
    def from_quat_xyzw(cls, t: np.ndarray, q_xyzw: np.ndarray) -> Pose:
        return cls(
            R=quat_xyzw_to_rotmat(q_xyzw),
            t=np.asarray(t, dtype=np.float64),
        )

    def as_matrix(self) -> np.ndarray:
        return make_transform(self.R, self.t)

    def as_quat_xyzw(self) -> np.ndarray:
        return rotmat_to_quat_xyzw(self.R)

    def inverse(self) -> Pose:
        R_inv = self.R.T
        t_inv = -R_inv @ self.t
        return Pose(R=R_inv, t=t_inv)

    def compose(self, other: Pose) -> Pose:
        """
        Compose two transforms.

        If:
            self  = T_A_B
            other = T_B_C

        then the result is:
            T_A_C
        """
        R = self.R @ other.R
        t = self.R @ other.t + self.t
        return Pose(R=R, t=t)

    def relative_to(self, reference: Pose) -> Pose:
        """
        Return this pose relative to another pose:

            T_reference_self = inv(T_W_reference) @ T_W_self
        """
        return reference.inverse().compose(self)

    def transform_points(self, points_C: np.ndarray) -> np.ndarray:
        """
        Transform points from the pose local frame to the parent frame.

        For T_W_C this means transforming points from C to W.
        """
        return transform_points(self.as_matrix(), points_C)

    def copy(self) -> Pose:
        return Pose(R=self.R.copy(), t=self.t.copy())


def interpolate_pose(pose0: Pose, pose1: Pose, alpha: float) -> Pose:
    """
    Interpolate two poses.

    Translation is interpolated linearly.
    Rotation is interpolated with quaternion SLERP.
    """
    alpha = float(alpha)

    t = (1.0 - alpha) * pose0.t + alpha * pose1.t

    q0 = pose0.as_quat_xyzw()
    q1 = pose1.as_quat_xyzw()
    q = slerp_quat_xyzw(q0, q1, alpha)

    return Pose.from_quat_xyzw(t=t, q_xyzw=q)
