# ============================================================
# Intelligent Dynamic Spectrum Cartography
# ROS 2 Humble + Gazebo Harmonic + ArduPilot SITL (Rover-4.5.7)
# Base: Ubuntu 22.04 (Jammy)
# ============================================================
FROM osrf/ros:humble-desktop

# ── System dependencies ──────────────────────────────────────
RUN apt-get update && apt-get install -y \
    wget gnupg lsb-release curl git build-essential \
    python3-pip python3-vcstool python3-colcon-common-extensions \
    default-jdk \
    # Mesa software rendering (no GPU required)
    xvfb x11vnc x11-apps \
    libgl1-mesa-dri libgl1-mesa-glx mesa-utils \
    libegl1 libgles2 libegl-dev \
    # (MAVProxy is not an Ubuntu apt package; it is installed with pip below)
 && rm -rf /var/lib/apt/lists/*

# ── Gazebo Harmonic (non-default pairing with ROS 2 Humble) ──
RUN wget https://packages.osrfoundation.org/gazebo.gpg \
      -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
 && echo "deb [arch=$(dpkg --print-architecture) \
      signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
      http://packages.osrfoundation.org/gazebo/ubuntu-stable \
      $(lsb_release -cs) main" \
      | tee /etc/apt/sources.list.d/gazebo-stable.list \
 && wget https://raw.githubusercontent.com/osrf/osrf-rosdep/master/gz/00-gazebo.list \
      -O /etc/ros/rosdep/sources.list.d/00-gazebo.list \
 && apt-get update && apt-get install -y gz-harmonic \
 && rm -rf /var/lib/apt/lists/*

# ── Python build deps ─────────────────────────────────────────
RUN pip install future mavproxy

# ── Non-root user (UID 1000 matches most Mac/Linux hosts) ────
ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
 && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
 && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME
USER $USERNAME
WORKDIR /home/$USERNAME

# ── microxrceddsgen (ArduPilot DDS code generator) ───────────
# Clone main branch (required for -default-container-prealloc-size support)
RUN git clone https://github.com/eProsima/Micro-XRCE-DDS-Gen.git \
 && cd Micro-XRCE-DDS-Gen \
 && git submodule update --init --recursive \
 && ./gradlew assemble

# ── ROS 2 workspace ───────────────────────────────────────────
RUN mkdir -p /home/$USERNAME/ros2_ws/src
WORKDIR /home/$USERNAME/ros2_ws

# Pull all repos defined by ardupilot_gz
RUN . /opt/ros/humble/setup.sh \
 && vcs import --input \
      https://raw.githubusercontent.com/ArduPilot/ardupilot_gz/main/ros2_gz.repos \
      --recursive src

# Pin ArduPilot to Rover-4.5.7 (stable, tested against microxrceddsgen 2.0.2)
RUN cd src/ardupilot \
 && git checkout Rover-4.5.7 \
 && git submodule update --init --recursive

# Patch 1: remove unsupported -default-container-prealloc-size flag
RUN sed -i \
  's/ -default-container-prealloc-size {container_prealloc_size}//' \
  src/ardupilot/libraries/AP_DDS/wscript

# Patch 2: switch Gazebo render engine from ogre2 → ogre
#          (ogre2 requires hardware GLX; ogre works with software Mesa)
RUN sed -i 's/ogre2/ogre/g' \
      src/ardupilot_gz/ardupilot_gz_gazebo/worlds/playpen.sdf

# Patch 3: create missing dds_use_ns.parm (not present in Rover-4.5.7)
# This is created at build time into the colcon install space post-build.
# A placeholder is staged here so the build doesn't fail.
RUN . /opt/ros/humble/setup.sh \
 && sudo apt-get update -q \
 && rosdep update \
 && rosdep install --from-paths src --ignore-src -y \
 && sudo rm -rf /var/lib/apt/lists/*

# ArduPilot build prerequisites the base image lacks
# (rsync for waf configure; Python modules from ArduPilot's install-prereqs list)
RUN sudo apt-get update -q \
 && sudo apt-get install -y rsync \
 && sudo rm -rf /var/lib/apt/lists/* \
 && sudo pip install pexpect ptyprocess empy==3.3.4 pymavlink pyserial \
      lxml dronecan future geocoder tabulate pyparsing psutil intelhex

# ── Build ─────────────────────────────────────────────────────
RUN . /opt/ros/humble/setup.sh \
 && export PATH=$PATH:/home/$USERNAME/Micro-XRCE-DDS-Gen/scripts \
 && export GZ_VERSION=harmonic \
 && colcon build \
      --packages-up-to ardupilot_gz_bringup \
      --executor sequential \
    || (cat log/latest_build/ardupilot_sitl/stdout_stderr.log; exit 1)

# Create the missing param file in the install space
RUN echo "DDS_ENABLE 1" > \
  /home/$USERNAME/ros2_ws/install/ardupilot_sitl/share/ardupilot_sitl/config/default_params/dds_use_ns.parm

# ── Environment ───────────────────────────────────────────────
RUN echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc \
 && echo 'source ~/ros2_ws/install/setup.bash 2>/dev/null || true' >> ~/.bashrc \
 && echo 'export PATH=$PATH:$HOME/Micro-XRCE-DDS-Gen/scripts' >> ~/.bashrc \
 && echo 'export GZ_VERSION=harmonic' >> ~/.bashrc \
 && echo 'export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:$HOME/ros2_ws/install/ardupilot_gz_bringup/share/ardupilot_gz_bringup/worlds:$HOME/ros2_ws/install/ardupilot_sitl_models/share/ardupilot_sitl_models/models:$HOME/ros2_ws/install/ardupilot_sitl_models/share/ardupilot_sitl_models/worlds' >> ~/.bashrc \
 && echo 'export SDF_PATH=$GZ_SIM_RESOURCE_PATH' >> ~/.bashrc \
 && echo 'export LIBGL_ALWAYS_SOFTWARE=1' >> ~/.bashrc \
 && echo 'export MESA_GL_VERSION_OVERRIDE=3.3' >> ~/.bashrc \
 && echo 'export MESA_LOADER_DRIVER_OVERRIDE=softpipe' >> ~/.bashrc \
 && echo 'export EGL_PLATFORM=surfaceless' >> ~/.bashrc

WORKDIR /home/$USERNAME/ros2_ws
CMD ["/bin/bash"]