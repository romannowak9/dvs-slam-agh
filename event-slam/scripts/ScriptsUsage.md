# Script usage

Run commands from the `event-slam` directory. The Docker environment described
in the repository `README.md` is the supported runtime.

## Complete experiments

These are the normal entry points for any correctly configured sequence:

```bash
./scripts/run_evslam.sh configs/my_evslam.yaml
./scripts/run_m3ed.sh configs/my_m3ed.yaml
```

Without an argument they use `configs/evslam.yaml` and `configs/m3ed.yaml`.
Dataset, reference, optional ground-truth and output paths are defined in YAML.
If ground truth is configured, the scripts also align to the first pose, create
plots and compute metrics. A ground-truth path passed as the second argument
overrides the value in YAML:

```bash
./scripts/run_evslam.sh configs/my_evslam.yaml /data/evSLAM/gt/my_gt.txt
./scripts/run_m3ed.sh configs/my_m3ed.yaml /data/m3ed/gt/my_gt.txt
```

When no ground truth is available, set `evaluation.ground_truth_path: null`.
The internal `config_run_paths.py` helper resolves output filenames for the
shell runners and is not normally called directly.

## Pipeline runner

Use the Python runner directly when working with a copied or modified config:

```bash
python3 scripts/run_event_slam.py --config configs/evslam.yaml
python3 scripts/run_event_slam.py --config configs/m3ed.yaml
```

The dataset reader is selected by `dataset.format`. A missing format means an
EvSLAM ROS bag; `m3ed_h5` selects the M3ED reader. All algorithm parameters and
output paths come from YAML.

## Alignment, plots and metrics

The final shell scripts contain the canonical commands. The tools can also be
called separately.

EvSLAM:

```bash
python3 scripts/align_evslam_result_to_gt.py \
  --estimate <output_dir>/<raw_result>.txt \
  --gt /data/evSLAM/gt/<ground_truth>.txt \
  --output <output_dir>/result_aligned_first_pose.txt \
  --method first_pose

python3 scripts/plot_trajectory.py \
  --trajectory <output_dir>/result_aligned_first_pose.txt \
  --gt /data/evSLAM/gt/<ground_truth>.txt \
  --output-dir <output_dir>/plots_first_pose \
  --prefix <sequence>_first_pose

python3 scripts/evaluate_evslam_metrics.py \
  --estimate <output_dir>/result_aligned_first_pose.txt \
  --gt /data/evSLAM/gt/<ground_truth>.txt \
  --output <output_dir>/metrics_first_pose.txt \
  --alignment none
```

M3ED:

```bash
python3 scripts/align_m3ed_result_to_gt.py \
  --estimate <output_dir>/<sequence>.txt \
  --gt /data/m3ed/gt/<ground_truth>.txt \
  --output <output_dir>/result_aligned_first_pose.txt \
  --method first_pose

python3 scripts/plot_m3ed_trajectory.py \
  --trajectory <output_dir>/result_aligned_first_pose.txt \
  --gt /data/m3ed/gt/<ground_truth>.txt \
  --output-dir <output_dir>/plots_first_pose \
  --prefix <sequence>_first_pose

python3 scripts/evaluate_m3ed_metrics.py \
  --estimate <output_dir>/result_aligned_first_pose.txt \
  --gt /data/m3ed/gt/<ground_truth>.txt \
  --output <output_dir>/metrics_first_pose.txt
```

`trajectory_metrics.py` contains shared metric functions and is imported by the
evaluators; it is not a command-line program.

## M3ED challenge package

For the three hidden challenge sequences, validate the result timestamps and
create the submission archive with:

```bash
python3 scripts/package_m3ed_submission.py \
  outputs/m3ed_challenge outputs/m3ed_submission.zip
```

Use `--reference-dir` if the timestamp files are not in `/data/m3ed/ref`.

## Diagnostic scripts

Diagnostic commands read a YAML config through `--config`. Use `--help` on a
specific script for optional preview limits and output switches.

Dataset inspection and event frames:

```bash
python3 scripts/verification/inspect_bag.py --config configs/evslam.yaml --inspect-imu
python3 scripts/verification/debug_event_windows.py --config configs/evslam.yaml --num-windows 10
python3 scripts/verification/debug_event_frames.py --config configs/evslam.yaml --save-preview
```

Calibration and IMU:

```bash
python3 scripts/verification/debug_calibration.py --config configs/evslam.yaml
python3 scripts/verification/debug_rectification.py --config configs/evslam.yaml --draw-lines
python3 scripts/verification/debug_imu_motion_compensation.py --config configs/evslam.yaml --save-preview
python3 scripts/verification/debug_imu_rotation.py --config configs/my_evslam.yaml --estimate <result.txt> --csv <imu_debug.csv>
```

Tracking, stereo depth and mapping:

```bash
python3 scripts/verification/debug_feature_tracking.py --config configs/evslam.yaml
python3 scripts/verification/debug_stereo_depth.py --config configs/evslam.yaml --max-draw-matches 200
python3 scripts/verification/debug_slam_map.py --config configs/evslam.yaml --max-plot-distance 5
python3 scripts/verification/plot_slam_map_results.py outputs/evslam
```

`verification_config.py` is a shared helper imported by the diagnostic scripts;
it is not a command-line program.
