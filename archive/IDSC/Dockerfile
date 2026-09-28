# ============================================================
# Dockerfile.sim  --  RF-source-seeking UGV simulation
# Dev laptop: x86_64 (amd64), NO GPU.
# Runtime = Gazebo Harmonic + ROS 2 Humble + CPU PyTorch.
# (No ArduPilot / Nav2 / MAVROS — the PPO policy is the planner.)
# Build:  docker build -f Dockerfile.sim -t ugv-sim .
# ============================================================
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8

# Prerequisites + UTF-8 locale
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gnupg lsb-release ca-certificates software-properties-common locales \
        python3-pip git \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# --- Gazebo Harmonic repository (OSRF) ---
RUN curl https://packages.osrfoundation.org/gazebo.gpg \
        --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
        | tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

# --- ROS 2 repository (current ros2-apt-source method) ---
RUN apt-get update \
    && ROS_APT_SOURCE_VERSION=$(curl -sI https://github.com/ros-infrastructure/ros-apt-source/releases/latest \
        | grep -i ^location: | awk -F'/' '{print $NF}' | tr -d '\r') \
    && if [ -z "$ROS_APT_SOURCE_VERSION" ]; then echo "Error: Failed to fetch ROS version"; exit 1; fi \
    && curl -fSL -o /tmp/ros2-apt-source.deb \
        "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo $VERSION_CODENAME)_all.deb" \
    && apt-get install -y /tmp/ros2-apt-source.deb \
    && rm -f /tmp/ros2-apt-source.deb \
    && rm -rf /var/lib/apt/lists/*

# --- ROS 2 + Gazebo Harmonic + the ROS<->Gazebo bridge + RViz ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-desktop \
        ros-dev-tools \
        ros-humble-rmw-cyclonedds-cpp \
        gz-harmonic \
        ros-humble-ros-gzharmonic \
    && rm -rf /var/lib/apt/lists/*

# CycloneDDS on loopback — reliable ROS 2 middleware under `docker --net host`
# (FastDDS, the default, hangs creating the DDS participant in this setup).
RUN echo '<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo" multicast="true"/></Interfaces></General></Domain></CycloneDDS>' > /etc/cyclonedds.xml
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV CYCLONEDDS_URI=file:///etc/cyclonedds.xml

# --- Software (CPU) OpenGL rendering -- REQUIRED on this GPU-less laptop ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-dri libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*
ENV LIBGL_ALWAYS_SOFTWARE=1

# --- Python ML runtime: CPU PyTorch + the pipeline's deps ---
# --ignore-installed: ROS pulled in distutils-managed sympy/numpy via apt, and
# pip cannot uninstall those. This installs fresh copies under /usr/local that
# shadow them instead of trying (and failing) to remove them.
# numpy pinned <2 to stay ABI-compatible with Humble's apt-built extensions.
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir --ignore-installed \
        torch --index-url https://download.pytorch.org/whl/cpu
RUN python3 -m pip install --no-cache-dir --ignore-installed \
        "numpy<2" scipy gymnasium pandas pyarrow matplotlib

# --- Non-root user (clean file ownership on the mounted workspace) ---
RUN groupadd --gid 1000 dev \
    && useradd --uid 1000 --gid 1000 -m -s /bin/bash dev
ENV HOME=/home/dev
USER dev
WORKDIR /home/dev

RUN echo "source /opt/ros/humble/setup.bash" >> /home/dev/.bashrc
CMD ["bash"]