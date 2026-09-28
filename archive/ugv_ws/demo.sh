#!/usr/bin/env bash
# demo.sh — run the RF-seeking UGV simulation (headless Gazebo + planner).
# Run inside the container, from ~/ugv_ws:   ./demo.sh
#
# Sets the CycloneDDS-on-loopback config that makes ROS node creation work under
# --net host, (re)generates the world if missing, then launches gz + bridge +
# the rf_seeker planner. Watch for "dist_to_emitter" ticking down.
set -e

export XDG_RUNTIME_DIR=/tmp/runtime-dev && mkdir -p "$XDG_RUNTIME_DIR"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo" multicast="true"/></Interfaces></General></Domain></CycloneDDS>'

export XDG_RUNTIME_DIR=/tmp/runtime-dev && mkdir -p "$XDG_RUNTIME_DIR"
# RMW_IMPLEMENTATION + CYCLONEDDS_URI are baked into the image (Dockerfile.sim).
# Do NOT set ROS_LOCALHOST_ONLY here: with the explicit CycloneDDS lo interface
# it causes "the same interface may not be selected twice".
unset ROS_LOCALHOST_ONLY

DATA="${1:-sample_28702.parquet}"

if [ ! -f rfsim.world ] || [ ! -f scene.json ]; then
  echo "[demo] generating world from $DATA ..."
  python3 make_world.py --data "$DATA" --decim 4
fi

echo "[demo] launching (gz headless + bridge + planner). Give it ~30-60s to load models."
ros2 launch rfsim.launch.py