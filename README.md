# Event-based stereo SLAM

This repository contains the event-based stereo SLAM system developed for a
master's thesis at AGH University of Krakow. It reads asynchronous events and
gyroscope measurements, forms stereo event frames, tracks features, estimates
metric motion, maintains a sparse map and optimizes loop-closure constraints.

The implementation supports two datasets:

- [IROS 2025 EvSLAM Challenge](https://nail-hnu.github.io/EvSLAM/data_format.html)
  (`.bag`, ROS messages),
- [M3ED](https://m3ed.io/) (`.h5`).

## Repository layout

```text
event-slam/
├── configs/              final EvSLAM and M3ED configurations
├── outputs/              selected results used in the thesis
├── scripts/              runners, evaluation and diagnostic tools
└── src/event_slam/       SLAM implementation
latex/mgr/                thesis sources and figures
event-slam/ros/dvs_msgs/  ROS message definitions required by EvSLAM bags
```

## From a fresh clone to results

The following procedure contains everything needed to run the system on a
selected sequence and obtain its outputs. Commands are run on a Linux host
with Git and Docker installed.

### 1. Clone the repository and build the image

```bash
git clone https://github.com/romannowak9/dvs-slam-agh.git
cd dvs-slam-agh
docker build -f event-slam/Dockerfile -t event-slam:thesis .
```

The image contains ROS Noetic and all required Python packages. No additional
Python environment or `requirements.txt` installation is needed.

### 2. Download one sequence and its supporting files

Create a dataset directory outside the repository, for example
`~/datasets`. You may use any sequence as long as all paths and ROS topics in
its copied configuration are updated.

For an EvSLAM sequence, download from the
[official EvSLAM page](https://nail-hnu.github.io/EvSLAM/data_format.html) or
its [Google Drive directory](https://drive.google.com/drive/folders/1Vjs0xYlBGE06t61pN1U1nTzKiWdGvGUv?usp=drive_link):

- the sequence ROS bag -- required;
- the camera calibration file for the platform used to record that sequence --
  required;
- the IMU calibration file -- required when `imu.enabled` is `true`;
- the sequence reference timestamps -- required to create a challenge-format
  result and to run the provided evaluation workflow;
- the ground-truth trajectory -- optional and used only for alignment, plots
  and metrics.

The calibration files contain YAML even if a downloaded copy uses a `.txt`
extension. Files may be arranged freely under the mounted dataset directory;
their locations are defined in the YAML configuration. For example:

```text
datasets/
└── evSLAM/
    ├── <sequence>.bag
    ├── calibration/
    │   ├── camera.yaml
    │   └── imu.yaml
    ├── ref/<sequence>_ref_time.txt
    └── gt/<sequence>_gt.txt          # optional
```

For an M3ED sequence, use the [official sequence list](https://m3ed.io/sequences/)
and [download instructions](https://m3ed.io/download/) to obtain:

- `<sequence>_data.h5` -- required; it contains events, IMU measurements and
  calibration used by this implementation;
- a text file with the required output timestamps -- required by the M3ED
  result writer;
- an EVO/TUM ground-truth pose file -- optional and used only for alignment,
  plots and metrics.

When ground truth is available, its first column can also serve as the required
timestamp list. For a sequence without published ground truth, use the
timestamp file supplied for that evaluation or challenge sequence.

```text
datasets/
└── m3ed/
    ├── <sequence>_data.h5
    ├── ref/<sequence>_timestamps.txt
    └── gt/<sequence>_pose_evo_gt.txt # optional
```

Depth files, semantic labels, videos and raw M3ED ROS bags are not read by the
pipeline.

### 3. Create a configuration for the selected sequence

Keep the committed files as reproducibility references and make a copy:

```bash
cp event-slam/configs/evslam.yaml event-slam/configs/my_evslam.yaml
# or
cp event-slam/configs/m3ed.yaml event-slam/configs/my_m3ed.yaml
```

Paths inside a configuration are container paths. The host dataset directory
will be mounted as `/data`, so a host file such as
`~/datasets/evSLAM/run01.bag` becomes `/data/evSLAM/run01.bag`.

Update these EvSLAM fields:

| YAML field | Meaning |
| --- | --- |
| `dataset.bag_path` | Path to the selected ROS bag. |
| `dataset.left_event_topic`, `dataset.right_event_topic` | Event topics present in that bag. |
| `dataset.camera_yaml` | Camera calibration matching the sequence platform. |
| `dataset.reference_timestamps_path` | Reference timestamp file used to write the result text file. |
| `imu.topic`, `imu.calibration_yaml` | IMU topic and matching calibration when IMU is enabled. |
| `evaluation.ground_truth_path` | Ground truth, or `null` when unavailable. |
| `output.output_dir`, `output.result_txt` | Unique output directory and raw result filename. |

Update these M3ED fields:

| YAML field | Meaning |
| --- | --- |
| `dataset.path` | Path to the selected `_data.h5` file. |
| `dataset.sequence_name` | Exact sequence name; it also determines the raw result filename. |
| `dataset.reference_timestamps_path` | Timestamp-only file or a ground-truth pose file containing the required timestamps. |
| `evaluation.ground_truth_path` | Ground truth, or `null` when unavailable. |
| `output.output_dir` | Unique directory for this run. |

The `processing.t_start`, `processing.t_end` and
`processing.num_frames` fields may optionally limit the processed range.
Leave them at `null`, `null` and `0` to process the complete sequence.

### 4. Run the selected sequence

From the repository root, set the absolute host path to the directory that
contains the dataset files:

```bash
DATASETS_DIR="$HOME/datasets"
```

Run an EvSLAM configuration:

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp \
  --volume "$PWD:/workspace" \
  --volume "$DATASETS_DIR:/data:ro" \
  event-slam:thesis \
  ./scripts/run_evslam.sh configs/my_evslam.yaml
```

Run an M3ED configuration:

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp \
  --volume "$PWD:/workspace" \
  --volume "$DATASETS_DIR:/data:ro" \
  event-slam:thesis \
  ./scripts/run_m3ed.sh configs/my_m3ed.yaml
```

If `evaluation.ground_truth_path` is set, the runner executes SLAM, aligns the
trajectory to the first ground-truth pose, creates plots and computes metrics.
An alternative ground-truth path can be passed as the second script argument.
For example, replace the command after `event-slam:thesis` in the corresponding
`docker run` invocation with one of these:

```bash
./scripts/run_evslam.sh configs/my_evslam.yaml /data/evSLAM/gt/my_gt.txt
./scripts/run_m3ed.sh configs/my_m3ed.yaml /data/m3ed/gt/my_gt.txt
```

If the ground-truth field is `null` and no second argument is supplied, the
runner executes SLAM and saves the raw outputs, then skips alignment, plots and
metrics.

### 5. Find the results

Results are written to the directory selected by `output.output_dir` and
remain on the host after the container exits. Every run contains:

```text
trajectory.csv       estimated camera trajectory
velocity.csv         estimated velocity
keyframes.csv        saved SLAM keyframes
landmarks.csv        saved sparse-map points
pose_graph.csv       saved pose-graph edges
```

EvSLAM also writes the filename selected by `output.result_txt`. M3ED writes
`<dataset.sequence_name>.txt`. When ground truth is configured, the runner
additionally creates:

```text
result_aligned_first_pose.txt   trajectory aligned to the first ground-truth pose
metrics_first_pose.txt          final evaluation report
plots_first_pose/               trajectory and velocity plots
```

Generated run directories are ignored by Git and do not overwrite the curated
results in `event-slam/outputs/thesis_results`.

## Native execution

Docker is the supported route. Native execution is possible on Ubuntu 20.04
with ROS Noetic, the `dvs_msgs` package built in a Catkin workspace, and the
Python packages listed in the Dockerfile. After sourcing ROS and the workspace:

```bash
source /opt/ros/noetic/setup.bash
source /path/to/catkin_ws/devel/setup.bash
cd event-slam
./scripts/run_evslam.sh configs/my_evslam.yaml
```

M3ED does not read ROS bags, but using the same container keeps numerical and
plotting dependencies consistent.

## Configuration and other tools

The committed YAML files are the final configurations used by the convenient
runners. Dataset paths, processing ranges and output names can be changed in a
copied YAML file. Low-level runner, alignment, plotting, evaluation and
diagnostic commands are documented in
[`event-slam/scripts/ScriptsUsage.md`](event-slam/scripts/ScriptsUsage.md).

Selected result artifacts underlying the tables and figures in the thesis are
stored in `event-slam/outputs/thesis_results`.
