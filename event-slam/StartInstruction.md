## Docker container

```bash
docker pull osrf/ros:noetic-desktop-full
```

Before opening container
```bash
xhost +local:docker
```

Running first time:
```bash
docker run -it \
  --name event-slam-dev \
  --net=host \
  --env="DISPLAY=$DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd):/workspace \
  -v /mnt/docker_disk/home/mgr/datasets:/data \
  -w /workspace \
  osrf/ros:noetic-desktop-full \
  bash
```

To delete this configuration
```bash
docker rm -f event-slam-dev
```

And later:
```bash
docker start -ai event-slam-dev
```

To open second terminal with the same container:
```bash
docker exec -it event-slam-dev bash
```

In VSCode attach to a running container and then open /workspace directory

### Quickstart

```bash
cd /mnt/docker_disk/home/mgr/dvs-slam-agh/event-slam
xhost +local:docker
docker start -ai event-slam-dev
```

Attach VSCode to running container and source:
```bash
source /opt/ros/noetic/setup.bash
```

## Usage of scripts

### Calibration debug script
```bash
python3 scripts/debug_calibration.py \
  --camera-yaml /data/evSLAM/drone_calibration/calib_results_cam_drone.yaml \
  --imu-yaml /data/evSLAM/drone_calibration/calib_results_imu_drone.yaml
```

### bag verification script
```bash
python3 scripts/inspect_bag.py \
  --bag /data/evSLAM/seq001.bag
```

or with all options
```bash
python3 scripts/inspect_bag.py \
  --bag /data/evSLAM/seq001.bag \
  --left-topic /dvxplorer_left/events \
  --right-topic /dvxplorer_right/events \
  --imu-topic /dvxplorer_left/imu \
  --inspect-imu \
  --num-batches 5 \
  --sample-events 8 \
  --full-scan
```

### Event window verification
```bash
python3 scripts/debug_event_windows.py \
  --bag /data/evSLAM/seq001.bag \
  --t-start 956.7
```

```bash
python3 scripts/debug_event_windows.py \
  --bag /data/evSLAM/seq001.bag \
  --left-topic /dvxplorer_left/events \
  --right-topic /dvxplorer_right/events \
  --time-window 0.0333333333 \
  --num-windows 5 \
  --sample-events 8 \
  --t-start 956.7 \
  --summary
```

### Event frames verification

```bash
python3 scripts/debug_event_frames.py \
  --bag /data/evSLAM/seq001.bag \
  --num-frames 10 \
  --time-window 0.1 \
  --save-preview
```

Exponential decay
```bash
python3 scripts/debug_event_frames.py \
  --bag /data/evSLAM/seq001.bag \
  --mode exponential \
  --tau 0.3 \
  --num-frames 10 \
  --time-window 0.3 \
  --save-preview
```

BAF Filter
```bash
python3 scripts/debug_event_frames.py \
  --bag /data/evSLAM/seq001.bag \
  --use-baf \
  --baf-time-window 0.0416667 \
  --num-frames 10 \
  --time-window 0.1 \
  --save-preview
```