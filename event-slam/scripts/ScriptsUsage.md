# Usage of scripts

## Calibration debug script
```bash
python3 scripts/verification/debug_calibration.py \
  --camera-yaml /data/evSLAM/drone_calibration/calib_results_cam_drone.yaml \
  --imu-yaml /data/evSLAM/drone_calibration/calib_results_imu_drone.yaml
```

## bag verification script
```bash
python3 scripts/verification/inspect_bag.py \
  --bag /data/evSLAM/seq001.bag
```

or with all options
```bash
python3 scripts/verification/inspect_bag.py \
  --bag /data/evSLAM/seq001.bag \
  --left-topic /dvxplorer_left/events \
  --right-topic /dvxplorer_right/events \
  --imu-topic /dvxplorer_left/imu \
  --inspect-imu \
  --num-batches 5 \
  --sample-events 8 \
  --full-scan
```

## Event window verification
```bash
python3 scripts/verification/debug_event_windows.py \
  --bag /data/evSLAM/seq001.bag \
  --t-start 956.7
```

```bash
python3 scripts/verification/debug_event_windows.py \
  --bag /data/evSLAM/seq001.bag \
  --left-topic /dvxplorer_left/events \
  --right-topic /dvxplorer_right/events \
  --time-window 0.0333333333 \
  --num-windows 5 \
  --sample-events 8 \
  --t-start 956.7 \
  --summary
```

## Event frames verification

```bash
python3 scripts/verification/debug_event_frames.py \
  --bag /data/evSLAM/seq001.bag \
  --num-frames 10 \
  --time-window 0.1 \
  --save-preview
```

Exponential decay
```bash
python3 scripts/verification/debug_event_frames.py \
  --bag /data/evSLAM/seq001.bag \
  --mode exponential \
  --tau 0.3 \
  --num-frames 10 \
  --time-window 0.3 \
  --save-preview
```

BAF Filter
```bash
python3 scripts/verification/debug_event_frames.py \
  --bag /data/evSLAM/seq001.bag \
  --mode exponential \
  --tau 0.004 \
  --use-baf \
  --baf-time-window 0.006 \
  --baf-radius 2 \
  --baf-min-neighbors 5 \
  --num-frames 10 \
  --time-window 0.006 \
  --t-start 970 \
  --save-preview
```

Use `--display` to display images in window and not save imgs to files

## Comparing sensors

Only color camera and left dvs
```bash
python3 scripts/verification/compare_sensors.py \
  --bag /data/evSLAM/seq001.bag \
  --mode exponential \
  --tau 0.03 \
  --use-baf \
  --baf-time-window 0.005 \
  --baf-radius 1 \
  --baf-min-neighbors 2 \
  --num-frames 100 \
  --time-window 0.005 \
  --t-start 950 \
  --only-color \
  --display
```

All left sensors
```bash
python3 scripts/verification/compare_sensors.py \
  --bag /data/evSLAM/seq001.bag \
  --mode exponential \
  --tau 0.004 \
  --use-baf \
  --baf-time-window 0.005 \
  --baf-radius 2 \
  --baf-min-neighbors 4 \
  --num-frames 10 \
  --time-window 0.005 \
  --t-start 970 \
  --display
```

Use `--display` to display images in window and not save imgs to files

## Stereo rectification verification

```bash
python3 scripts/verification/debug_rectification.py \
  --bag /data/evSLAM/seq001.bag \
  --camera-yaml /data/evSLAM/drone_calibration/calib_results_cam_drone.yaml \
  --num-frames 10 \
  --time-window 0.005 \
  --t-start 970 \
  --draw-lines
```

or

```bash
python3 scripts/verification/debug_rectification.py \
  --bag /data/evSLAM/seq001.bag \
  --camera-yaml /data/evSLAM/drone_calibration/calib_results_cam_drone.yaml \
  --mode exponential \
  --tau 0.007 \
  --use-baf \
  --baf-time-window 0.007 \
  --baf-radius 2 \
  --baf-min-neighbors 5 \
  --time-window 0.007 \
  --t-start 970 \
  --num-frames 10 \
  --draw-lines
```

Use `--display` to display images in window and not save imgs to files

## Feature tracker verification
```bash
python3 scripts/verification/debug_feature_tracking.py \
  --bag /data/evSLAM/seq001.bag \
  --camera-yaml /data/evSLAM/drone_calibration/calib_results_cam_drone.yaml \
  --t-start 970 \
  --num-frames 100 \
  --time-window 0.005 \
  --display
```

or

```bash
python3 scripts/verification/debug_feature_tracking.py \
  --bag /data/evSLAM/seq001.bag \
  --camera-yaml /data/evSLAM/drone_calibration/calib_results_cam_drone.yaml \
  --mode exponential \
  --tau 0.007 \
  --time-window 0.007 \
  --use-baf \
  --baf-time-window 0.005 \
  --baf-radius 2 \
  --baf-min-neighbors 5 \
  --detector gftt \
  --t-start 970 \
  --num-frames 10
```

## Stereo depth estimation - triangulation

```bash
python3 scripts/verification/debug_stereo_depth.py \
  --bag /data/evSLAM/seq001.bag \
  --camera-yaml /data/evSLAM/drone_calibration/calib_results_cam_drone.yaml \
  --t-start 970 \
  --num-frames 100 \
  --display
```

```bash
python3 scripts/verification/debug_stereo_depth.py \
  --bag /data/evSLAM/seq001.bag \
  --camera-yaml /data/evSLAM/drone_calibration/calib_results_cam_drone.yaml \
  --mode exponential \
  --tau 0.006 \
  --time-window 0.007 \
  --use-baf \
  --baf-time-window 0.005 \
  --baf-radius 2 \
  --baf-min-neighbors 5 \
  --max-draw-matches 40 \
  --detector gftt \
  --epipolar-threshold 2.0 \
  --min-disparity 0.5 \
  --max-disparity 250 \
  --min-depth 0.05 \
  --max-depth 100 \
  --t-start 970 \
  --num-frames 200 \
  --display
```

## PnP Visual Odometry

```bash
python3 scripts/verification/run_stereo_pnp_vo.py \
  --bag /data/evSLAM/seq001.bag \
  --camera-yaml /data/evSLAM/drone_calibration/calib_results_cam_drone.yaml \
  --t-start 970 \
  --num-frames 200 \
  --display
```

or

```bash
python3 scripts/verification/run_stereo_pnp_vo.py \
  --bag /data/evSLAM/seq001.bag \
  --camera-yaml /data/evSLAM/drone_calibration/calib_results_cam_drone.yaml \
  --mode exponential \
  --tau 0.006 \
  --time-window 0.007 \
  --use-baf \
  --baf-time-window 0.005 \
  --baf-radius 2 \
  --baf-min-neighbors 5 \
  --t-start 970 \
  --num-frames 100 \
  --display
```

To save trajectory of entire bag:
```bash
python3 scripts/verification/run_stereo_pnp_vo.py \
  --bag /data/evSLAM/seq001.bag \
  --camera-yaml /data/evSLAM/drone_calibration/calib_results_cam_drone.yaml \
  --mode exponential \
  --tau 0.01 \
  --time-window 0.01 \
  --detector gftt \
  --use-baf \
  --baf-time-window 0.007 \
  --baf-radius 2 \
  --baf-min-neighbors 5 \
  --num-frames 0 \
  --display
```

## Align results to gt

```bash
python3 scripts/align_evslam_result_to_gt.py \
  --estimate outputs/evslam_vo_seq007_try03/result_seq007.txt \
  --gt /data/evSLAM/gt/seq007_test_gt.txt \
  --output outputs/evslam_vo_seq007_try03/result_seq007_aligned_se3.txt \
  --method se3  # or first_pose or sim3
```

Or with only first sample

```bash
python3 scripts/align_evslam_result_to_gt.py \
  --estimate outputs/evslam_vo_seq007_/result_seq007.txt \
  --gt /data/evSLAM/gt/seq007_test_gt.txt \
  --output outputs/evslam_vo_seq007_/result_seq007_aligned_first_pose.txt \
  --method first_pose
```

## Plot trajectory

```bash
python3 scripts/plot_trajectory.py \
  --trajectory outputs/evslam_vo_seq001_v02/result_seq001.txt \
  --output-dir outputs/evslam_vo_seq001_v02/plots
```

Or with gt:

```bash
python3 scripts/plot_trajectory.py \
  --trajectory outputs/evslam_vo_seq007_rot_comp_full/result_seq007_aligned_first_pose.txt \
  --output-dir outputs/evslam_vo_seq007_rot_comp_full/plots_first_pose \
  --gt /data/evSLAM/gt/seq007_test_gt.txt
```

## Evaluate EvSLAM metrics: ATE i AUC

metrics described at website: https://nail-hnu.github.io/EvSLAM/competition.html

```bash
python3 scripts/evaluate_evslam_metrics.py \
  --estimate outputs/evslam_vo_seq007/result_seq007.txt \
  --gt /data/evSLAM/gt/seq007_test_gt.txt \
  --output outputs/evslam_vo_seq007/metrics.txt
```

## IMU rotation verification

```bash
python3 scripts/verification/debug_imu_rotation.py \
  --bag /data/evSLAM/seq007_test.bag \
  --camera-calibration /data/evSLAM/mecanum_calibration/calib_results_cam_others.yaml \
  --imu-calibration /data/evSLAM/mecanum_calibration/calib_results_imu_others.yaml \
  --estimate outputs/evslam_vo_seq007_/result_seq007_aligned_first_pose.txt \
  --csv outputs/evslam_vo_seq007_/imu_rotation_debug.csv
```

## Event windows rotation compensation with IMU

```bash
python3 scripts/verification/debug_imu_motion_compensation.py \
  --bag /data/evSLAM/seq001.bag \
  --camera-calibration /data/evSLAM/drone_calibration/calib_results_cam_drone.yaml \
  --imu-calibration /data/evSLAM/drone_calibration/calib_results_imu_drone.yaml \
  --mode exponential \
  --tau 0.02 \
  --use-baf \
  --baf-time-window 0.01 \
  --baf-radius 2 \
  --baf-min-neighbors 5 \
  --num-frames 10 \
  --time-window 0.02 \
  --t-start 970 \
  --save-preview \
  --output-dir outputs/debug_imu_motion_compensation
```

```bash
python3 scripts/verification/debug_imu_motion_compensation.py \
  --bag /data/evSLAM/seq007_test.bag \
  --camera-calibration /data/evSLAM/mecanum_calibration/calib_results_cam_others.yaml \
  --imu-calibration /data/evSLAM/mecanum_calibration/calib_results_imu_others.yaml \
  --mode exponential \
  --tau 0.05 \
  --use-baf \
  --baf-time-window 0.01 \
  --baf-radius 2 \
  --baf-min-neighbors 5 \
  --num-frames 10 \
  --time-window 0.05 \
  --save-preview \
  --output-dir outputs/debug_imu_motion_compensation_seq007
```

Use `--display` instead of `--save-preview` for an interactive preview.

## Sparse SLAM map verification

Save every keyframe with its landmark observations and create a 3D map plot:

```bash
python3 scripts/verification/debug_slam_map.py \
  --config configs/evslam_seq007_test_slam.yaml \
  --num-frames 120
```

Green points are landmarks created in the displayed keyframe. Yellow points
are landmarks already present in an earlier keyframe. Add `--display` to show
the keyframes while the pipeline is running. Images are saved in
`outputs/debug_slam_map` by default.
