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

## Usage of scripts

Calibration debug script
```bash
python3 scripts/debug_calibration.py \
  --camera-yaml /data/evSLAM/drone_calibration/calib_results_cam_drone.yaml \
  --imu-yaml /data/evSLAM/drone_calibration/calib_results_imu_drone.yaml
```