FROM osrf/ros:noetic-desktop-full

ENV DEBIAN_FRONTEND=noninteractive
ENV nproc=8

RUN apt update && apt upgrade -y

RUN apt install -y \
    build-essential  \
    cmake \
    pkg-config  \
    htop  \
    gedit  \
    wget \
    git \
    unzip  \
    curl \
    vim \
    gedit \
    software-properties-common \
    libboost-all-dev \
    net-tools \
    iputils-ping \
    libeigen3-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    ros-noetic-geodesy \
    ros-noetic-nmea-msgs \
    libqglviewer-dev-qt5 \
    && rm -rf /var/lib/apt/lists/*
    
# ceres
WORKDIR /opt
RUN wget -O ceres.tar.gz http://ceres-solver.org/ceres-solver-2.1.0.tar.gz
RUN tar -xzvf ceres.tar.gz -C /opt && rm ceres.tar.gz
WORKDIR /opt/ceres-solver-2.1.0
RUN mkdir build && cd build && cmake .. && make -j$(nproc) && make install
RUN rm -rf /opt/ceres-solver-2.1.0

WORKDIR /opt
# Pinned. g2o's master renames types between releases -- this tree depends on the
# Isometry3 / Vector7 spelling -- so an unpinned clone builds a different library
# every time and eventually stops compiling. 20241228_git is the last tag before
# the image this workspace was developed against.
ARG G2O_TAG=20241228_git
RUN git clone --depth 1 --branch ${G2O_TAG} https://github.com/RainerKuemmerle/g2o.git /opt/g2o
WORKDIR /opt/g2o
RUN mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc) && make install
RUN rm -rf /opt/g2o

WORKDIR /root/catkin_ws

#RUN git clone https://github.com/wooseongY/Go-RIO
#WORKDIR /root/catkin_ws
#RUN . /opt/ros/noetic/setup.sh && catkin_make
#RUN rm -rf /var/lib/apt/lists/*
#RUN echo source /opt/ros/noetic/setup.bash >> /root/.bashrc
#RUN echo source /root/catkin_ws/devel/setup.bash >> /root/.bashrc

# The ROS 1 apt signing key shipped in the base image expired (EXPKEYSIG
# F42ED6FBAB17C654, Open Robotics), which makes any apt-get update fail with
# exit 100. Refresh the keyring the repo actually points at via signed-by.
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc \
      | gpg --dearmor > /usr/share/keyrings/ros1-latest-archive-keyring.gpg

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# evo lives in its own virtualenv on purpose. Installing it into the system
# interpreter pulls numpy 1.17 -> 1.24 and replaces matplotlib, and numpy 1.24
# dropped the np.float / np.int / np.bool aliases that ROS Noetic's Python
# packages still use. The venv keeps the ROS interpreter untouched.
# Version pinned to what pip resolves for Python 3.8, for reproducibility.
RUN python3 -m venv /opt/evo-venv \
    && /opt/evo-venv/bin/pip install --no-cache-dir --upgrade "pip<25" setuptools wheel \
    && /opt/evo-venv/bin/pip install --no-cache-dir "evo==1.31.1"

# Expose evo_ape / evo_rpe / evo_traj on PATH without shadowing system Python.
RUN for b in /opt/evo-venv/bin/evo*; do ln -sf "$b" "/usr/local/bin/$(basename "$b")"; done

# viser: browser-based visualization, replacing RViz for both live monitoring and
# post-hoc result review. Lives in its own venv created with
# --system-site-packages so it can import rospy and the ROS message packages,
# with numpy force-installed at 1.24 inside the venv because viser needs
# numpy.typing (>=1.20) while the system interpreter must keep numpy 1.17 for
# ROS Noetic's own Python packages. Verified that numpy 1.24.4, viser 1.1.0,
# rospy, nav_msgs, sensor_msgs.point_cloud2 and tf.transformations all import
# and work together in this venv.
RUN python3 -m venv --system-site-packages /opt/viser-venv \
    && /opt/viser-venv/bin/pip install --no-cache-dir --upgrade "pip<25" setuptools wheel \
    && /opt/viser-venv/bin/pip install --no-cache-dir --ignore-installed \
         "numpy==1.24.4" "viser==1.1.0" "scipy==1.10.1"

# g2o installs into /usr/local/lib but the base image never ran ldconfig, so the
# runtime linker cannot find libg2o_stuff.so.0.2 and every nodelet fails to
# dlopen. ROS only puts /opt/ros/noetic/lib on LD_LIBRARY_PATH.
RUN ldconfig && ldconfig -p | grep -q libg2o_stuff || \
    (echo "g2o still not on the linker path" >&2; exit 1)

WORKDIR /root/catkin_ws
