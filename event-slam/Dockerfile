FROM osrf/ros:noetic-desktop-full

SHELL ["/bin/bash", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-h5py \
        python3-matplotlib \
        python3-numpy \
        python3-opencv \
        python3-scipy \
        python3-yaml \
    && rm -rf /var/lib/apt/lists/*

COPY event-slam/ros/dvs_msgs /tmp/dvs_ws/src/dvs_msgs
RUN source /opt/ros/noetic/setup.bash \
    && catkin_make -C /tmp/dvs_ws install -DCMAKE_INSTALL_PREFIX=/opt/ros/noetic \
    && rm -rf /tmp/dvs_ws

ENV MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /workspace/event-slam

CMD ["bash"]
