from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from event_slam.core.geometry import Pose, rotmat_to_quat_xyzw
from event_slam.core.trajectory import Trajectory
from event_slam.core.velocity import VelocityTrajectory, world_to_camera_velocities
from event_slam.debug.visualization import format_value


@dataclass
class ResultWriterStats:
    """
    Statistics returned after writing a challenge result file.
    """

    reference_count: int
    written_count: int
    skipped_count: int
    first_written_timestamp: float = np.nan
    last_written_timestamp: float = np.nan


def save_outputs(pipeline, config: dict, interrupted: bool = False) -> tuple:
    """Save all configured pipeline outputs and print the run report."""
    output_cfg = config.get("output", {})
    velocity_cfg = config.get("velocity", {})
    dataset_cfg = config.get("dataset", {})
    output_dir = Path(output_cfg.get("output_dir", "outputs/evslam_vo"))
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory_path = _resolve_output_path(
        output_dir,
        output_cfg.get("trajectory_csv", "trajectory.csv"),
    )
    write_vo_csv(pipeline.slam.results, trajectory_path)

    keyframes_path = None
    landmarks_path = None
    pose_graph_path = None
    sparse_map = pipeline.slam.map
    if sparse_map is not None:
        keyframes_path = _resolve_output_path(
            output_dir,
            output_cfg.get("keyframes_csv", "keyframes.csv"),
        )
        landmarks_path = _resolve_output_path(
            output_dir,
            output_cfg.get("landmarks_csv", "landmarks.csv"),
        )
        write_keyframes_csv(sparse_map.keyframes, keyframes_path)
        write_landmarks_csv(sparse_map.landmarks.values(), landmarks_path)
        pose_graph_path = _resolve_output_path(
            output_dir,
            output_cfg.get("pose_graph_csv", "pose_graph.csv"),
        )
        write_pose_graph_csv(pipeline.slam.pose_graph.edges, pose_graph_path)

    velocity_path = None
    if bool(velocity_cfg.get("enabled", False)):
        velocity = pipeline.compute_velocity()
        velocity_path = _resolve_output_path(
            output_dir,
            velocity_cfg.get("output_csv", "velocity.csv"),
        )
        write_velocity_csv(velocity, velocity_path)

    result_path = None
    result_stats = None
    dataset_format = dataset_cfg.get("format", "evslam_rosbag")
    reference_path = dataset_cfg.get("reference_timestamps_path")
    if dataset_format == "m3ed_h5" and reference_path is None:
        raise ValueError("M3ED output requires dataset.reference_timestamps_path")
    if reference_path and len(pipeline.trajectory) == 0 and dataset_format != "m3ed_h5":
        print("Skipping challenge result: trajectory is empty.")
    elif reference_path:
        if dataset_format == "m3ed_h5":
            sequence_name = dataset_cfg.get("sequence_name")
            if not sequence_name:
                raise ValueError("M3ED output requires dataset.sequence_name")
            result_path = output_dir / f"{sequence_name}.txt"
            result_stats = write_m3ed_result_from_reference_file(
                trajectory=pipeline.trajectory,
                reference_path=reference_path,
                output_path=result_path,
                sequence_name=sequence_name,
                allow_partial=interrupted,
            )
        else:
            result_path = _resolve_output_path(
                output_dir,
                output_cfg.get("result_txt", "result.txt"),
            )
            result_stats = write_result_from_reference_file(
                trajectory=pipeline.trajectory,
                velocity=pipeline.velocity_trajectory,
                reference_path=reference_path,
                output_path=result_path,
                skip_out_of_range=True,
            )

    _print_summary(pipeline.get_summary())
    _print_saved_outputs(
        trajectory_path,
        velocity_path,
        result_path,
        result_stats,
        keyframes_path,
        landmarks_path,
        pose_graph_path,
    )

    if interrupted:
        print()
        print("Partial outputs saved after KeyboardInterrupt.")

    return trajectory_path, velocity_path, result_path, result_stats


def write_vo_csv(results, path) -> None:
    """Write the trajectory and VO diagnostics CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        file.write(
            "timestamp,success,tx,ty,tz,qx,qy,qz,qw,"
            "track_count,pnp_point_count,pnp_inlier_count,"
            "reprojection_error_mean,reprojection_error_median,"
            "pnp_rotation_step_deg,imu_rotation_step_deg,"
            "pnp_imu_rotation_error_deg,imu_rotation_consistent,"
            "imu_rejected,message,pose_source,map_point_count,"
            "map_inlier_count,local_landmark_count,"
            "map_descriptor_match_count,map_message,new_feature_count,"
            "tracking_state,reference_keyframe_id,is_keyframe,"
            "loop_candidate_count,loop_candidate_id,loop_match_count,"
            "loop_accepted,relocalized,graph_cost_before,graph_cost_after,"
            "loop_relative_scale,local_ba_success,local_ba_cost_before,"
            "local_ba_cost_after,local_ba_landmark_count,"
            "local_ba_visual_cost_before,local_ba_visual_cost_after,"
            "local_ba_imu_cost_before,local_ba_imu_cost_after,"
            "local_ba_evaluations,local_ba_imu_factor_count\n"
        )

        for result in results:
            T_W_C = result.T_W_Cleft
            t = T_W_C[:3, 3]
            qx, qy, qz, qw = rotmat_to_quat_xyzw(T_W_C[:3, :3])
            file.write(
                f"{result.timestamp:.9f},"
                f"{int(result.success)},"
                f"{t[0]:.9f},{t[1]:.9f},{t[2]:.9f},"
                f"{qx:.9f},{qy:.9f},{qz:.9f},{qw:.9f},"
                f"{result.track_count},"
                f"{result.pnp_point_count},"
                f"{result.pnp_inlier_count},"
                f"{result.reprojection_error_mean:.9f},"
                f"{result.reprojection_error_median:.9f},"
                f"{result.pnp_rotation_step_deg:.9f},"
                f"{result.imu_rotation_step_deg:.9f},"
                f"{result.pnp_imu_rotation_error_deg:.9f},"
                f"{int(result.imu_rotation_consistent)},"
                f"{int(result.imu_rejected)},"
                f"{_csv_safe(result.message)},"
                f"{result.pose_source},"
                f"{result.map_point_count},"
                f"{result.map_inlier_count},"
                f"{result.local_landmark_count},"
                f"{result.map_descriptor_match_count},"
                f"{_csv_safe(result.map_message)},"
                f"{result.new_feature_count},"
                f"{result.tracking_state},"
                f"{result.reference_keyframe_id},"
                f"{int(result.is_keyframe)},"
                f"{result.loop_candidate_count},"
                f"{result.loop_candidate_id},"
                f"{result.loop_match_count},"
                f"{int(result.loop_accepted)},"
                f"{int(result.relocalized)},"
                f"{result.graph_cost_before:.9f},"
                f"{result.graph_cost_after:.9f},"
                f"{result.loop_relative_scale:.9f},"
                f"{int(result.local_ba_success)},"
                f"{result.local_ba_cost_before:.9f},"
                f"{result.local_ba_cost_after:.9f},"
                f"{result.local_ba_landmark_count},"
                f"{result.local_ba_visual_cost_before:.9f},"
                f"{result.local_ba_visual_cost_after:.9f},"
                f"{result.local_ba_imu_cost_before:.9f},"
                f"{result.local_ba_imu_cost_after:.9f},"
                f"{result.local_ba_evaluations},"
                f"{result.local_ba_imu_factor_count}\n"
            )


def write_velocity_csv(velocity: VelocityTrajectory, path) -> None:
    """Write camera-frame velocity samples to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        velocity.as_camera_array(),
        fmt="%.9f",
        delimiter=",",
        header="timestamp,vx_camera,vy_camera,vz_camera,speed",
        comments="",
    )


def write_keyframes_csv(keyframes, path) -> None:
    """Write keyframe poses and map point counts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        file.write(
            "id,frame_index,timestamp,tx,ty,tz,qx,qy,qz,qw,point_count,"
            "vx_O,vy_O,vz_O,bgx,bgy,bgz,bax,bay,baz\n"
        )
        for keyframe in keyframes:
            t = keyframe.T_W_C[:3, 3]
            qx, qy, qz, qw = rotmat_to_quat_xyzw(keyframe.T_W_C[:3, :3])
            file.write(
                f"{keyframe.id},{keyframe.frame_index},{keyframe.timestamp:.9f},"
                f"{t[0]:.9f},{t[1]:.9f},{t[2]:.9f},"
                f"{qx:.9f},{qy:.9f},{qz:.9f},{qw:.9f},"
                f"{keyframe.point_count},"
                + ",".join(
                    f"{value:.9f}"
                    for value in np.concatenate(
                        (
                            keyframe.velocity_O,
                            keyframe.gyro_bias,
                            keyframe.accel_bias,
                        )
                    )
                )
                + "\n"
            )


def write_landmarks_csv(landmarks, path) -> None:
    """Write sparse map landmarks and observation counts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        file.write(
            "id,anchor_keyframe_id,x_anchor,y_anchor,z_anchor,"
            "x_world,y_world,z_world,observation_count,last_seen_keyframe_id,"
            "inverse_depth,depth_uncertainty\n"
        )
        for landmark in sorted(landmarks, key=lambda item: item.id):
            x_C = landmark.position_C_anchor
            x_W = landmark.position_W
            file.write(
                f"{landmark.id},{landmark.anchor_keyframe_id},"
                f"{x_C[0]:.9f},{x_C[1]:.9f},{x_C[2]:.9f},"
                f"{x_W[0]:.9f},{x_W[1]:.9f},{x_W[2]:.9f},"
                f"{landmark.observation_count},"
                f"{landmark.last_seen_keyframe_id},"
                f"{landmark.inverse_depth:.9f},"
                f"{landmark.depth_uncertainty:.9f}\n"
            )


def write_pose_graph_csv(edges, path) -> None:
    """Write sequential and loop pose-graph constraints."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        file.write(
            "source_id,target_id,type,tx,ty,tz,qx,qy,qz,qw,"
            "inlier_count,reprojection_error_median\n"
        )
        for edge in edges:
            T = edge.T_C_source_C_target
            t = T[:3, 3]
            qx, qy, qz, qw = rotmat_to_quat_xyzw(T[:3, :3])
            file.write(
                f"{edge.source_id},{edge.target_id},{edge.edge_type},"
                f"{t[0]:.9f},{t[1]:.9f},{t[2]:.9f},"
                f"{qx:.9f},{qy:.9f},{qz:.9f},{qw:.9f},"
                f"{edge.inlier_count},{edge.reprojection_error_median:.9f}\n"
            )


def load_evslam_result_array(path) -> np.ndarray:
    """
    Load an EvSLAM 11-column result file.

    Expected columns:
        timestamp tx ty tz qx qy qz qw vx vy vz
    """
    path = Path(path)
    rows = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.replace(",", " ").split()

            if len(parts) != 11:
                raise ValueError(
                    f"Expected 11 columns in {path} at line {line_number}, "
                    f"got {len(parts)}: {line}"
                )

            try:
                rows.append([float(value) for value in parts])
            except ValueError as exc:
                raise ValueError(
                    f"Could not parse numeric values in {path} "
                    f"at line {line_number}: {line}"
                ) from exc

    if not rows:
        raise ValueError(f"No samples found in file: {path}")

    return np.asarray(rows, dtype=np.float64)


def load_evslam_result(path) -> tuple[Trajectory, VelocityTrajectory]:
    """
    Load EvSLAM poses and camera-frame velocities as project trajectories.
    """
    data = load_evslam_result_array(path)
    trajectory = Trajectory()
    velocity = VelocityTrajectory()

    for row in data:
        pose = Pose.from_quat_xyzw(t=row[1:4], q_xyzw=row[4:8])
        velocity_camera = row[8:11]

        trajectory.append(timestamp=float(row[0]), pose=pose)
        velocity.append(
            timestamp=float(row[0]),
            velocity_world=pose.R @ velocity_camera,
            velocity_camera=velocity_camera,
        )

    return trajectory, velocity


def load_reference_timestamps(path) -> np.ndarray:
    """
    Load reference timestamps from a text file.

    Empty lines and comment lines starting with '#' are ignored. If the file has
    multiple columns, the first column is interpreted as the timestamp.
    """
    path = Path(path)
    timestamps = []

    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            try:
                timestamps.append(float(parts[0]))
            except (IndexError, ValueError) as exc:
                raise ValueError(
                    f"Could not parse timestamp in {path} at line {line_number}: {line}"
                ) from exc

    return np.asarray(timestamps, dtype=np.float64)


def write_trajectory_at_timestamps(
    trajectory: Trajectory,
    velocity: VelocityTrajectory,
    timestamps: np.ndarray,
    output_path,
    skip_out_of_range: bool = True,
) -> ResultWriterStats:
    """
    Interpolate a trajectory at selected timestamps and save challenge output.

    Output format:
        timestamp tx ty tz qx qy qz qw vx vy vz

    The linear velocity is expressed in the current camera frame.
    """
    if trajectory.is_empty:
        raise ValueError("Cannot write result from an empty trajectory")

    if velocity.is_empty:
        raise ValueError("Cannot write result from an empty velocity trajectory")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)

    interpolated, skipped_count = _interpolate_valid_poses(
        trajectory=trajectory,
        timestamps=timestamps,
        skip_out_of_range=skip_out_of_range,
    )

    velocities_world = _interpolate_world_velocities(
        velocity=velocity,
        timestamps=interpolated.timestamps,
    )
    velocities_camera = world_to_camera_velocities(
        trajectory=interpolated,
        velocities_world=velocities_world,
    )

    written_count = 0
    first_written_timestamp = np.nan
    last_written_timestamp = np.nan

    with open(output_path, "w", encoding="utf-8") as file:
        for pose_sample, velocity_camera in zip(
            interpolated.samples,
            velocities_camera,
        ):
            timestamp = pose_sample.timestamp
            pose = pose_sample.pose

            tx, ty, tz = pose.t
            qx, qy, qz, qw = rotmat_to_quat_xyzw(pose.R)
            vx, vy, vz = velocity_camera

            file.write(
                f"{timestamp:.9f} "
                f"{tx:.9f} {ty:.9f} {tz:.9f} "
                f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f} "
                f"{vx:.9f} {vy:.9f} {vz:.9f}\n"
            )

            if written_count == 0:
                first_written_timestamp = float(timestamp)

            last_written_timestamp = float(timestamp)
            written_count += 1

    return ResultWriterStats(
        reference_count=len(timestamps),
        written_count=written_count,
        skipped_count=skipped_count,
        first_written_timestamp=first_written_timestamp,
        last_written_timestamp=last_written_timestamp,
    )


def write_result_from_reference_file(
    trajectory: Trajectory,
    velocity: VelocityTrajectory,
    reference_path,
    output_path,
    skip_out_of_range: bool = True,
) -> ResultWriterStats:
    """
    Load reference timestamps and write the trajectory interpolated to them.
    """
    timestamps = load_reference_timestamps(reference_path)

    return write_trajectory_at_timestamps(
        trajectory=trajectory,
        velocity=velocity,
        timestamps=timestamps,
        output_path=output_path,
        skip_out_of_range=skip_out_of_range,
    )


def write_m3ed_result_from_reference_file(
    trajectory: Trajectory,
    reference_path,
    output_path,
    sequence_name: str,
    allow_partial: bool = False,
) -> ResultWriterStats:
    """Write M3ED output, optionally limited to the available trajectory range."""
    timestamps = load_reference_timestamps(reference_path)
    _validate_reference_timestamps(timestamps)
    output_path = Path(output_path)
    expected_name = f"{sequence_name}.txt"
    if output_path.name != expected_name:
        raise ValueError(
            f"M3ED result must be named {expected_name}, got {output_path.name}"
        )
    if trajectory.is_empty:
        raise ValueError("Cannot write M3ED result from an empty trajectory")
    covered = (
        (timestamps >= trajectory.first().timestamp)
        & (timestamps <= trajectory.last().timestamp)
    )
    if not allow_partial and not np.all(covered):
        raise ValueError(
            "M3ED reference range is outside trajectory coverage: "
            f"[{timestamps[0]:.9f}, {timestamps[-1]:.9f}] not within "
            f"[{trajectory.first().timestamp:.9f}, {trajectory.last().timestamp:.9f}]"
        )
    selected_timestamps = timestamps[covered] if allow_partial else timestamps
    if len(selected_timestamps) == 0:
        raise ValueError("Trajectory does not cover any M3ED reference timestamp")

    data = trajectory_to_m3ed_array(trajectory.interpolate_many(
        selected_timestamps,
        clamp=False,
    ))
    _validate_m3ed_array(data, selected_timestamps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, data, fmt="%.9f")
    validate_m3ed_result_file(
        output_path,
        sequence_name=sequence_name,
        reference_path=None if allow_partial else reference_path,
    )
    return ResultWriterStats(
        reference_count=len(timestamps),
        written_count=len(data),
        skipped_count=len(timestamps) - len(data),
        first_written_timestamp=float(selected_timestamps[0]),
        last_written_timestamp=float(selected_timestamps[-1]),
    )


def trajectory_to_m3ed_array(trajectory: Trajectory) -> np.ndarray:
    """Convert internal T_W_C poses to the orientation convention used by M3ED."""
    data = trajectory.as_tum_array()
    data[:, 4:7] *= -1.0
    return data


def trajectory_from_m3ed_array(data: np.ndarray) -> Trajectory:
    """Convert M3ED positions and R_C_W quaternions to internal T_W_C poses."""
    data = np.asarray(data, dtype=np.float64).copy()
    data[:, 4:7] *= -1.0
    return Trajectory.from_tum_array(data)


def validate_m3ed_result_file(
    path,
    sequence_name: str = None,
    reference_path=None,
) -> np.ndarray:
    """Validate one M3ED challenge file and optionally its reference times."""
    path = Path(path)
    if sequence_name is not None and path.name != f"{sequence_name}.txt":
        raise ValueError(f"Expected M3ED result {sequence_name}.txt, got {path.name}")
    try:
        data = np.loadtxt(path, dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read M3ED result: {path}") from exc
    if data.ndim == 1:
        data = data.reshape(1, -1)

    reference_timestamps = None
    if reference_path is not None:
        reference_timestamps = load_reference_timestamps(reference_path)
        _validate_reference_timestamps(reference_timestamps)
    _validate_m3ed_array(data, reference_timestamps)
    return data


def _validate_reference_timestamps(timestamps: np.ndarray) -> None:
    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if len(timestamps) == 0 or not np.all(np.isfinite(timestamps)):
        raise ValueError("M3ED reference timestamps must be non-empty and finite")
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("M3ED reference timestamps must be strictly increasing")


def _validate_m3ed_array(
    data: np.ndarray,
    reference_timestamps: np.ndarray = None,
) -> None:
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 8 or len(data) == 0:
        raise ValueError(f"M3ED result must be a non-empty Nx8 array, got {data.shape}")
    if not np.all(np.isfinite(data)):
        raise ValueError("M3ED result contains non-finite values")
    if np.any(np.diff(data[:, 0]) <= 0.0):
        raise ValueError("M3ED result timestamps must be strictly increasing")
    quaternion_norms = np.linalg.norm(data[:, 4:8], axis=1)
    if not np.allclose(quaternion_norms, 1.0, atol=1e-6):
        raise ValueError("M3ED result quaternions must be non-zero and normalized")
    if reference_timestamps is not None and not np.array_equal(
        data[:, 0], np.asarray(reference_timestamps, dtype=np.float64)
    ):
        raise ValueError("M3ED result timestamps do not exactly match the reference")


def _interpolate_valid_poses(
    trajectory: Trajectory,
    timestamps: np.ndarray,
    skip_out_of_range: bool,
) -> tuple:
    """
    Interpolate trajectory samples and optionally skip out-of-range timestamps.
    """
    output = Trajectory()
    skipped_count = 0

    for timestamp in timestamps:
        timestamp = float(timestamp)

        try:
            pose = trajectory.interpolate(timestamp, clamp=False)
        except ValueError as exc:
            if skip_out_of_range:
                skipped_count += 1
                continue

            raise ValueError(
                f"Reference timestamp {timestamp:.9f} is outside trajectory range"
            ) from exc

        output.append(timestamp=timestamp, pose=pose)

    return output, skipped_count


def _interpolate_world_velocities(
    velocity: VelocityTrajectory,
    timestamps: np.ndarray,
) -> np.ndarray:
    """
    Linearly interpolate world-frame velocity to selected timestamps.
    """
    source_timestamps = velocity.timestamps
    source_velocities = velocity.velocities_world
    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)

    if np.any(np.diff(source_timestamps) <= 0.0):
        raise ValueError("Velocity timestamps must be strictly increasing")

    if len(timestamps) == 0:
        return np.empty((0, 3), dtype=np.float64)

    if timestamps[0] < source_timestamps[0] or timestamps[-1] > source_timestamps[-1]:
        raise ValueError("Requested timestamps are outside velocity range")

    return np.column_stack(
        [
            np.interp(timestamps, source_timestamps, source_velocities[:, axis])
            for axis in range(3)
        ]
    ).astype(np.float64)


def _print_summary(summary) -> None:
    print()
    print("SLAM summary")
    print("=" * 80)
    print(f"processed_frames: {summary.processed_frames}")
    print(f"successful_steps: {summary.successful_steps}")
    print(f"failed_frames: {summary.failed_frames}")
    print(f"median_inliers: {format_value(summary.median_inliers)}")
    print(f"velocity_samples: {summary.velocity_samples}")
    print(f"motion_compensated_frames: {summary.motion_compensated_frames}")
    print(f"motion_compensation_failed: {summary.motion_compensation_failed}")
    print(f"imu_prior_available_frames: {summary.imu_prior_available_frames}")
    print(f"imu_rejected_steps: {summary.imu_rejected_steps}")
    print(f"keyframes: {summary.keyframe_count}")
    print(f"landmarks: {summary.landmark_count}")
    print(f"accepted_loops: {summary.accepted_loop_count}")
    print(f"relocalizations: {summary.relocalization_count}")
    print(f"graph_cost_before: {format_value(summary.graph_cost_before)}")
    print(f"graph_cost_after: {format_value(summary.graph_cost_after)}")
    print(
        "final_position: "
        f"[{summary.final_position[0]:.6f}, "
        f"{summary.final_position[1]:.6f}, "
        f"{summary.final_position[2]:.6f}]"
    )


def _print_saved_outputs(
    trajectory_path,
    velocity_path,
    result_path,
    result_stats,
    keyframes_path,
    landmarks_path,
    pose_graph_path,
) -> None:
    print()
    print("Saved outputs")
    print("=" * 80)
    print(f"trajectory_csv: {trajectory_path}")

    if velocity_path is not None:
        print(f"velocity_csv: {velocity_path}")

    if result_path is not None:
        print(f"result_txt: {result_path}")
        if result_stats is not None:
            print(
                "result_stats: "
                f"reference={result_stats.reference_count}, "
                f"written={result_stats.written_count}, "
                f"skipped={result_stats.skipped_count}"
            )

    if keyframes_path is not None:
        print(f"keyframes_csv: {keyframes_path}")
        print(f"landmarks_csv: {landmarks_path}")
        print(f"pose_graph_csv: {pose_graph_path}")


def _resolve_output_path(output_dir: Path, path_value) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else output_dir / path


def _csv_safe(value: str) -> str:
    return str(value).replace(",", ";").replace("\n", " ")
