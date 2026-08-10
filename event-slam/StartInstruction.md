# Docker container

```bash
docker pull osrf/ros:noetic-desktop-full
```

Before opening container
```bash
xhost +local:docker
```

Running first time:
```bash
cd /mnt/docker_disk/home/mgr/dvs-slam-agh

docker run -it \
  --name event-slam-dev \
  --net=host \
  --security-opt seccomp=unconfined \
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

## Quickstart

```bash
cd /mnt/docker_disk/home/mgr/dvs-slam-agh/event-slam
xhost +local:docker
docker start -ai event-slam-dev
```

Attach VSCode to running container and source:
```bash
source /opt/ros/noetic/setup.bash
cd /workspace/event-slam
apt-get install -y --no-install-recommends python3-pip
python3 -m pip install --no-deps scipy==1.7.3
```

`--no-deps` keeps the container's NumPy 1.17.4 unchanged.

# Trajectory estimation

```bash
python3 scripts/run_evslam_slam.py \
  --config configs/evslam_seq007_test_slam.yaml
```
