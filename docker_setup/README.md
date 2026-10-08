# Intelligent Dynamic Spectrum Cartography
### Simulation Environment — ROS 2 Humble · Gazebo Harmonic · ArduRover SITL

---

## Overview

This repository contains the full autonomous ground vehicle (AGV) simulation stack for the Intelligent Dynamic Spectrum Cartography project. The simulation environment runs **ArduRover 4.5.7 SITL** connected to **Gazebo Harmonic** through **ROS 2 Humble** via DDS, all inside a reproducible Docker container that works on macOS (Apple Silicon and Intel), Windows 11, and native Linux.

The container provides:
- ROS 2 Humble (Ubuntu 22.04 base)
- Gazebo Harmonic with software rendering (no GPU required)
- ArduPilot SITL pinned to `Rover-4.5.7`
- Full ROS ↔ Gazebo ↔ ArduPilot DDS bridge
- All sensor topics: GPS, IMU, magnetometer, LiDAR, odometry, battery

---

## Prerequisites

Install these on your **host machine** before anything else.

| Tool | macOS | Windows 11 | Linux |
|------|-------|------------|-------|
| [Docker Desktop](https://www.docker.com/products/docker-desktop/) | ✅ | ✅ | use Docker Engine |
| [VS Code](https://code.visualstudio.com/) | ✅ | ✅ | ✅ |
| VS Code [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) | ✅ | ✅ | ✅ |
| [XQuartz](https://www.xquartz.org/) (X11 server) | ✅ | ❌ | ❌ |
| [VcXsrv](https://sourceforge.net/projects/vcxsrv/) or [Xming](https://sourceforge.net/projects/xming/) | ❌ | ✅ | ❌ |

### Docker Desktop settings (macOS and Windows)
Open Docker Desktop → Settings → Resources and set:
- **Memory:** 8 GB minimum (12 GB recommended)
- **CPUs:** 4 minimum
- **Disk:** 30 GB minimum (the build is large)

---

## Quick Start

### Step 1 — Clone the repository

```bash
git clone https://github.com/Senior-Design-Spectrum-Cartography/Intelligent-Motion-Planning.git
cd Intelligent-Motion-Planning
```

### Step 2 — Open in VS Code

```bash
code .
```

VS Code will detect the `.devcontainer/` folder and show a popup:
**"Reopen in Container"** — click it.

If the popup doesn't appear: `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Win/Linux) → **Dev Containers: Reopen in Container**

> **First build takes 30–60 minutes.** Docker is compiling ArduPilot from source and building the full ROS workspace. Subsequent opens are instant (cached layers).

### Step 3 — Set up X11 display (before launching the sim)

**macOS only — run in a regular Mac Terminal (not VS Code):**
```bash
open -a XQuartz
xhost +127.0.0.1
```
Do this every time you restart your Mac or Docker Desktop.

**Windows only — run XcXsrv with these settings:**
- Multiple windows
- Display number: 0
- Start no client
- Disable access control

Open Windows Security $\rarr$ "Allow an app through firewall"  
* Find VcXsrv windows xserver
* Enable Private and Public

### Step 4 — Launch the simulation

Open a terminal inside the VS Code Dev Container (`Terminal → New Terminal`) and run:

```bash
bash  
~/launch_sim.sh
```

Wait for this line before proceeding:
```
[mavproxy.py] AP: ArduPilot Ready
```
This takes approximately 30 seconds.

### Step 5 — Verify the simulation is running

Open a **second terminal** inside the container and run:

```bash
source ~/ros2_ws/install/setup.bash
ros2 topic list
```

You should see:
```
/battery
/clock
/gpsfix
/gz/tf
/gz/tf_static
/imu
/joint_states
/magnetometer
/navsat
/odometry
/scan
/tf
/tf_static
```

Confirm data is flowing:
```bash
gz topic -e -t /world/playpen/model/wildthumper/link/base_link/sensor/navsat_sensor/navsat 2>/dev/null | head -5
```

---

## Visualization with RViz

In the second terminal, start the VNC server and RViz:

```bash
export DISPLAY=host.docker.internal:0
rviz2 -d ~/ros2_ws/install/ardupilot_gz_bringup/share/ardupilot_gz_bringup/rviz/wildthumper.rviz &
```

**macOS:** Finder → Go → Connect to Server → `vnc://localhost:5900`  
**Windows:** Open any VNC viewer → connect to `localhost:5900`

---

## Repository Structure

```
Intelligent-Motion-Planning/
├── .devcontainer/
│   └── devcontainer.json        # VS Code Dev Container config
├── Dockerfile                   # Full environment definition
├── launch_sim.sh                # One-shot simulation launcher
├── README.md                    # This file
├── jetson_radiomap/             # TensorRT deployment scripts (Workstream 1)
│   ├── models_radiomap.py
│   ├── pth_to_onnx.py
│   ├── onnx_to_engine.py
│   └── ablation_inference_benchmark.py
└── ros2_ws/                     # ROS 2 workspace (built inside container)
    └── src/                     # Cloned by vcs import during Docker build
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Docker Container (Ubuntu 22.04)                        │
│                                                         │
│  ┌──────────────┐    JSON/UDP    ┌──────────────────┐  │
│  │ ArduRover    │◄──────────────►│ Gazebo Harmonic  │  │
│  │ SITL v4.5.7  │                │ (playpen world)  │  │
│  └──────┬───────┘                └──────┬───────────┘  │
│         │ DDS (UDP port 2019)           │ gz-transport  │
│         ▼                               ▼               │
│  ┌──────────────┐             ┌──────────────────────┐  │
│  │ micro-ROS    │             │ ros_gz_bridge        │  │
│  │ Agent        │             │ (sensor topics)      │  │
│  └──────┬───────┘             └──────────┬───────────┘  │
│         └─────────────┬──────────────────┘              │
│                       ▼                                 │
│              ROS 2 Topic Bus                            │
│         /navsat /imu /scan /odometry ...                │
│                       │                                 │
│              [Your nodes go here]                       │
│         Spectrum reconstruction · PPO planner           │
│         YOLO11n · Object avoidance pipeline             │
└─────────────────────────────────────────────────────────┘
```

---

## Key Engineering Decisions

### Why Docker instead of native macOS/Windows install?
ROS 2 Humble + Gazebo Harmonic is only officially supported on Ubuntu 22.04. Native macOS support exists but requires patching dozens of CMake files and is brittle. Docker gives every team member an identical Ubuntu 22.04 environment in one command.

### Why ArduPilot `Rover-4.5.7` instead of `master`?
ArduPilot `master` (as of mid-2025) introduced namespace-scoped C++ constants in the DDS layer (`GlobalPosition::IGNORE_LATITUDE` etc.) that require a version of `microxrceddsgen` that does not yet have a stable release tag. `Rover-4.5.7` is the latest stable release that builds cleanly against `microxrceddsgen` v2.0.2.

### Why software rendering (`LIBGL_ALWAYS_SOFTWARE=1`)?
Docker Desktop on macOS and Windows does not expose a physical GPU or a compatible GLX framebuffer to containers. Gazebo Harmonic's sensor system (IMU, NavSat, LiDAR) requires an OpenGL context to initialize its render pipeline. Mesa's software rasterizer (`softpipe`) satisfies this requirement without hardware.

### Why `ogre` instead of `ogre2`?
Gazebo Harmonic defaults to `ogre2` (Ogre-Next), which requires OpenGL 3.3+ with hardware acceleration. The `ogre` (v1) renderer works with Mesa's software stack and provides equivalent sensor simulation capability for ground vehicle applications. The sky rendering system is unavailable under `ogre` (non-critical for this project).

### Why Xvfb?
Even in headless mode, Gazebo's sensor system opens an X display connection during render context initialization. Xvfb provides a virtual framebuffer that satisfies this requirement without a physical display or XQuartz GLX passthrough (which fails in Docker Desktop's virtualization layer).

---

## Manual Build (advanced)

If you prefer to build the workspace yourself rather than using the pre-built Docker image:

```bash
# Inside the container terminal
cd ~/ros2_ws

# 1. Pull all repos
vcs import --input \
  https://raw.githubusercontent.com/ArduPilot/ardupilot_gz/main/ros2_gz.repos \
  --recursive src

# 2. Pin ArduPilot to stable release
cd src/ardupilot
git checkout Rover-4.5.7
git submodule update --init --recursive
cd ~/ros2_ws

# 3. Apply required patches
sed -i 's/ -default-container-prealloc-size {container_prealloc_size}//' \
  src/ardupilot/libraries/AP_DDS/wscript

sed -i 's/ogre2/ogre/g' \
  src/ardupilot_gz/ardupilot_gz_gazebo/worlds/playpen.sdf

# 4. Install dependencies
source /opt/ros/humble/setup.bash
rosdep update
rosdep install --from-paths src --ignore-src -y

# 5. Build
export PATH=$PATH:$HOME/Micro-XRCE-DDS-Gen/scripts
colcon build --packages-up-to ardupilot_gz_bringup --executor sequential

# 6. Create missing param file
echo "DDS_ENABLE 1" > \
  install/ardupilot_sitl/share/ardupilot_sitl/config/default_params/dds_use_ns.parm
```

---

## Troubleshooting

**`source: not found` when running commands**  
You are in `/bin/sh` instead of bash. Run `bash` first, or prefix commands with `bash -c "..."`.

**`Could not find the program ['microxrceddsgen']`**  
The generator is not on PATH. Run:
```bash
export PATH=$PATH:$HOME/Micro-XRCE-DDS-Gen/scripts
microxrceddsgen -version
```

**`PANIC: Failed to load defaults from ...dds_use_ns.parm`**  
The param file is missing. Run:
```bash
echo "DDS_ENABLE 1" > \
  ~/ros2_ws/install/ardupilot_sitl/share/ardupilot_sitl/config/default_params/dds_use_ns.parm
```

**`Unable to create a suitable GLXContext`**  
The software rendering environment is not set. Ensure these are exported before launching:
```bash
export DISPLAY=:99
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3
export MESA_LOADER_DRIVER_OVERRIDE=softpipe
```

**`/navsat` topic not publishing data**  
The Gazebo sensor render thread may be stuck. Check:
```bash
gz topic -l | grep navsat
gz topic -e -t /world/playpen/model/wildthumper/link/base_link/sensor/navsat_sensor/navsat 2>/dev/null | head -5
```

**Build killed with `cc1plus: Killed signal`**  
Docker Desktop ran out of memory during compilation. Increase Docker Desktop memory to 8+ GB in Settings → Resources.

---

## Workstream Status

| Workstream | Status | Notes |
|---|---|---|
| **WS1 — Model deployment** |  Complete | CNN, WNet, PartialConvMAE ablation; TensorRT engines built on Jetson Orin Nano |
| **WS2 — Simulation harness** |  In progress | ROS 2 + Gazebo + ArduPilot SITL running; sensor data pipeline under debugging |
| **WS3 — Autonomy integration** |  Pending WS2 | Spectrum reconstruction node, PPO planner, YOLO11n, obstacle avoidance |
| **WS4 — Hardware construction** |  Parts arriving | Power budget done; Jetson image ready; bring-up checklist written |

---

## Related Documentation

- [ArduPilot ROS 2 with Gazebo](https://ardupilot.org/dev/docs/ros2-gazebo.html)
- [ardupilot_gz repository](https://github.com/ArduPilot/ardupilot_gz)
- [Gazebo Harmonic + ROS 2 Humble (non-default pairing)](https://gazebosim.org/docs/harmonic/ros_installation)
- [VS Code Dev Containers with ROS 2](https://docs.ros.org/en/humble/How-To-Guides/Setup-ROS-2-with-VSCode-and-Docker-Container.html)
