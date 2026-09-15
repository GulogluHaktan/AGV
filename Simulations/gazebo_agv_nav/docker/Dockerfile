FROM osrf/ros:jazzy-desktop-full

RUN apt-get update && apt-get install -y \
    ros-jazzy-ros-gz \
    ros-jazzy-turtlebot3 \
    ros-jazzy-turtlebot3-msgs \
    ros-jazzy-turtlebot3-gazebo \
    python3-pip \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --break-system-packages \
    gymnasium stable-baselines3 torch hydra-core omegaconf matplotlib

# arm 3 (equivariant policy): E(2)-steerable CNN library. Needs gfortran
# (above) to build py3nj/lie_learn. --ignore-installed avoids pip trying (and
# failing) to uninstall the debian-packaged scipy.
RUN pip3 install --break-system-packages --ignore-installed escnn

ENV TURTLEBOT3_MODEL=burger
# GZ Transport UDP multicast discovery is broken under this host's Docker
# --network host setup without an explicit bind IP; without this, gz-sim and
# ros_gz_bridge can't discover each other's topics (ROS->GZ publish works,
# but GZ->ROS subscriptions like /odom silently receive nothing).
ENV GZ_IP=127.0.0.1
WORKDIR /workspace
