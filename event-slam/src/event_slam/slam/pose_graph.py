from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from event_slam.core.geometry import (
    invert_transform,
    se3_exp,
    se3_log,
)


@dataclass
class PoseGraphEdge:
    source_id: int
    target_id: int
    T_C_source_C_target: np.ndarray
    edge_type: str
    inlier_count: int = 0
    reprojection_error_median: float = np.nan


@dataclass
class PoseGraphOptimization:
    poses: list
    cost_before: float
    cost_after: float
    evaluations: int
    success: bool


class PoseGraph:
    """Small SE(3) keyframe graph with sequential and loop constraints."""

    def __init__(
        self,
        translation_weight: float = 1.0,
        rotation_weight: float = 1.0,
        huber_scale: float = 1.0,
        max_evaluations: int = 100,
    ) -> None:
        self.edges = []
        self._inverse_measurements = []
        self.weights = np.array(
            [translation_weight] * 3 + [rotation_weight] * 3,
            dtype=np.float64,
        )
        self.huber_scale = float(huber_scale)
        self.max_evaluations = int(max_evaluations)

    def add_edge(
        self,
        source_id: int,
        target_id: int,
        T_C_source_C_target: np.ndarray,
        edge_type: str,
        inlier_count: int = 0,
        reprojection_error_median: float = np.nan,
    ) -> PoseGraphEdge:
        edge = PoseGraphEdge(
            source_id=int(source_id),
            target_id=int(target_id),
            T_C_source_C_target=np.asarray(
                T_C_source_C_target, dtype=np.float64
            ).copy(),
            edge_type=str(edge_type),
            inlier_count=int(inlier_count),
            reprojection_error_median=float(reprojection_error_median),
        )
        self.edges.append(edge)
        self._inverse_measurements.append(
            invert_transform(edge.T_C_source_C_target)
        )
        return edge

    def optimize(self, keyframes) -> PoseGraphOptimization:
        initial = [keyframe.T_W_C.copy() for keyframe in keyframes]
        if len(initial) < 2 or not self.edges:
            return PoseGraphOptimization(initial, 0.0, 0.0, 0, True)

        x0 = np.zeros(6 * (len(initial) - 1), dtype=np.float64)
        residual_before = self._residuals(x0, initial)
        result = least_squares(
            self._residuals,
            x0,
            args=(initial,),
            jac_sparsity=self._jacobian_sparsity(len(initial)),
            loss="huber",
            f_scale=self.huber_scale,
            max_nfev=self.max_evaluations,
        )
        residual_after = self._residuals(result.x, initial)
        return PoseGraphOptimization(
            poses=self._apply_increments(result.x, initial),
            cost_before=self._huber_cost(residual_before),
            cost_after=self._huber_cost(residual_after),
            evaluations=int(result.nfev),
            success=bool(result.success),
        )

    def _residuals(self, increments: np.ndarray, initial: list) -> np.ndarray:
        poses = self._apply_increments(increments, initial)
        inverse_poses = [invert_transform(pose) for pose in poses]
        residuals = []
        for edge, inverse_measurement in zip(
            self.edges,
            self._inverse_measurements,
        ):
            predicted = inverse_poses[edge.source_id] @ poses[edge.target_id]
            error = inverse_measurement @ predicted
            residuals.append(self.weights * se3_log(error))
        return np.concatenate(residuals)

    def _jacobian_sparsity(self, pose_count: int):
        """Return the two-pose block structure of the graph Jacobian."""
        variable_count = 6 * (pose_count - 1)
        sparsity = lil_matrix(
            (6 * len(self.edges), variable_count),
            dtype=np.int8,
        )
        for edge_index, edge in enumerate(self.edges):
            rows = slice(6 * edge_index, 6 * (edge_index + 1))
            for pose_id in (edge.source_id, edge.target_id):
                if pose_id > 0:
                    cols = slice(6 * (pose_id - 1), 6 * pose_id)
                    sparsity[rows, cols] = 1
        return sparsity.tocsr()

    @staticmethod
    def _apply_increments(increments: np.ndarray, initial: list) -> list:
        poses = [initial[0].copy()]
        for index, pose in enumerate(initial[1:]):
            delta = increments[6 * index : 6 * (index + 1)]
            poses.append(pose @ se3_exp(delta))
        return poses

    def _huber_cost(self, residuals: np.ndarray) -> float:
        absolute = np.abs(residuals)
        scale = self.huber_scale
        losses = np.where(
            absolute <= scale,
            residuals**2,
            2.0 * scale * absolute - scale**2,
        )
        return 0.5 * float(np.sum(losses))
