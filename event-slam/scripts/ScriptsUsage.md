# Scripts usage

Verification scripts use the existing project YAML for dataset and algorithm
settings. Parameters already present in YAML, such as event topics, processing
range, aggregation, BAF, rectification, tracker, stereo depth, PnP, IMU and
`output.output_dir`, are not repeated in the command line.

Options used only by a particular diagnostic remain CLI arguments. Use
`--help` to list them. The default configuration is
`configs/evslam_seq007_test_slam.yaml`; pass another one with `--config`.

## Dataset and event processing

```bash
python3 scripts/verification/inspect_bag.py \
  --config configs/evslam_seq007_test_slam.yaml \
  --inspect-imu

python3 scripts/verification/debug_event_windows.py \
  --config configs/evslam_seq007_test_slam.yaml \
  --num-windows 10

python3 scripts/verification/debug_event_frames.py \
  --config configs/evslam_seq007_test_slam.yaml \
  --save-preview

python3 scripts/verification/compare_sensors.py \
  --config configs/evslam_seq007_test_slam.yaml \
  --only-color
```

`height`, `width`, camera topics used only for sensor comparison, display
options and inspection limits remain optional arguments.

## Calibration, rectification and IMU

```bash
python3 scripts/verification/debug_calibration.py \
  --config configs/evslam_seq007_test_slam.yaml

python3 scripts/verification/debug_rectification.py \
  --config configs/evslam_seq007_test_slam.yaml \
  --draw-lines

python3 scripts/verification/debug_imu_motion_compensation.py \
  --config configs/evslam_seq007_test_slam.yaml \
  --save-preview

python3 scripts/verification/debug_imu_rotation.py \
  --config configs/evslam_seq007_test_slam.yaml \
  --estimate outputs/example/result_seq007.txt \
  --csv outputs/example/imu_rotation_debug.csv
```

The diagnostic-only `extra-timeshift`, preview and display settings remain
arguments.

## Tracking, depth and pose estimation

```bash
python3 scripts/verification/debug_feature_tracking.py \
  --config configs/evslam_seq007_test_slam.yaml

python3 scripts/verification/debug_stereo_depth.py \
  --config configs/evslam_seq007_test_slam.yaml \
  --max-draw-matches 200

python3 scripts/verification/run_stereo_pnp_vo.py \
  --config configs/evslam_seq007_test_vo.yaml \
  --save-debug
```

## SLAM map

Run SLAM and save keyframe images plus map diagnostics:

```bash
python3 scripts/verification/debug_slam_map.py \
  --config configs/evslam_seq007_test_slam.yaml \
  --max-plot-distance 5
```

Create only `map_3d.png` and `tracking_diagnostics.png` from saved CSV files:

```bash
python3 scripts/verification/plot_slam_map_results.py \
  outputs/evslam_seq007_slam_stage_2_full
```

## Result analysis

These parameters are not part of the algorithm YAML and therefore remain CLI
arguments:

```bash
python3 scripts/align_evslam_result_to_gt.py \
  --estimate outputs/example/result_seq007.txt \
  --gt /data/evSLAM/gt/seq007_test_gt.txt \
  --output outputs/example/result_seq007_aligned_se3.txt \
  --method se3

python3 scripts/plot_trajectory.py \
  --trajectory outputs/example/result_seq007_aligned_se3.txt \
  --gt /data/evSLAM/gt/seq007_test_gt.txt \
  --output-dir outputs/example/plots

python3 scripts/evaluate_evslam_metrics.py \
  --estimate outputs/example/result_seq007_aligned_se3.txt \
  --gt /data/evSLAM/gt/seq007_test_gt.txt \
  --output outputs/example/metrics.txt
```

M3ED uses separate analysis scripts because its challenge files contain poses
only (`timestamp tx ty tz qx qy qz qw`). The plotting script derives smoothed
camera-frame velocities from both estimated and GT poses. It displays both
trajectories directly in the official M3ED camera frame, without reordering
axes. M3ED positions describe the camera in the initial-camera frame, while the
published quaternion represents the inverse rotation; the M3ED scripts perform
this conversion at the file boundary and use project-standard `T_W_C`
internally.

```bash
python3 scripts/align_m3ed_result_to_gt.py \
  --estimate outputs/example/falcon_outdoor_day_fast_flight_2.txt \
  --gt /data/m3ed/gt/falcon_outdoor_day_fast_flight_2_pose_evo_gt.txt \
  --output outputs/example/result_aligned_se3.txt \
  --method se3

python3 scripts/plot_m3ed_trajectory.py \
  --trajectory outputs/example/result_aligned_se3.txt \
  --gt /data/m3ed/gt/falcon_outdoor_day_fast_flight_2_pose_evo_gt.txt \
  --output-dir outputs/example/plots_se3

python3 scripts/evaluate_m3ed_metrics.py \
  --estimate outputs/example/result_aligned_se3.txt \
  --gt /data/m3ed/gt/falcon_outdoor_day_fast_flight_2_pose_evo_gt.txt \
  --output outputs/example/metrics_se3.txt
```

## Main SLAM runner

```bash
python3 scripts/run_event_slam.py \
  --config configs/evslam_seq007_test_slam.yaml
```

The same runner selects M3ED from `dataset.format: m3ed_h5`:

```bash
python3 scripts/run_event_slam.py \
  --config configs/m3ed_falcon_fast_flight_2_slam.yaml
```

An interrupted M3ED run still saves the trajectory, map diagnostics and a
partial `<sequence_name>.txt`. The partial challenge file contains only the
reference timestamps covered by the trajectory; a completed run remains strict
and must cover every reference timestamp.

M3ED requires the system package `python3-h5py` documented in
`StartInstruction.md`.

Inspect reader throughput without running SLAM:

```bash
python3 scripts/verification/inspect_m3ed_h5.py
```

Validate and package the three official challenge files:

```bash
python3 scripts/package_m3ed_submission.py \
  outputs/m3ed_challenge outputs/m3ed_submission.zip
```

Debug images are written below `output.output_dir` from the selected YAML. Use
a copied configuration with a different output directory when you want to keep
them separate from the main run.
