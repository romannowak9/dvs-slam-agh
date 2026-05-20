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
  --name pose-est-dev \
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
docker rm -f pose-est-dev
```

And later:
```bash
docker start -ai pose-est-dev
```

To open second terminal with the same container:
```bash
docker exec -it pose-est-dev bash
```

In VSCode attach to a running container and then open /workspace directory

## ROS

```bash
cd catkin_ws
source devel/setup.bash
```

### After changes in code
```bash
catkin_make
```

### Launch
Firstly:
```bash
roscore
```

Play dataset. For example:
```bash
rosbag play /data/evSLAM/seq001.bag
```

To visualize imu:
```bash
rqt_plot /dvxplorer_left/imu/linear_acceleration/x:y:z
```

### Quickstart

```bash
cd /mnt/docker_disk/home/mgr/dvs-slam-agh/pose-estimation
xhost +local:docker
docker start -ai pose-est-dev
roscore
```

```bash
roslaunch pose_estimation pose_estimation.launch
```

