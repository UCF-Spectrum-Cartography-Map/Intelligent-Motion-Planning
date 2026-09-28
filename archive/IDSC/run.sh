#!/usr/bin/env bash
# Launch the UGV simulation container with GUI (X11) + a persistent workspace.
# Assumes a Linux host running X11.
#
#   ./run.sh          # start the container (first shell)
#   ./run.sh exec     # open ANOTHER shell into the running container
#
set -e

IMAGE=ugv-sim
CONTAINER=ugv_sim
WS="$HOME/ugv_ws"     # your code lives here on the HOST and survives container restarts

# Open an extra shell into the already-running container (for multi-process launches)
if [ "$1" = "exec" ]; then
  exec docker exec -it --env GZ_IP=127.0.0.1 --env ROS_LOCALHOST_ONLY=1 "$CONTAINER" bash
fi

mkdir -p "$WS"
xhost +local:docker >/dev/null    # let the container talk to your X server

docker run -it --rm \
  --name "$CONTAINER" \
  --add-host "$(hostname):127.0.0.1" \
  --net host \
  --ipc host \
  --env DISPLAY="$DISPLAY" \
  --env LIBGL_ALWAYS_SOFTWARE=1 \
  --env GZ_IP=127.0.0.1 \
  --env ROS_LOCALHOST_ONLY=1 \
  --env QT_X11_NO_MITSHM=1 \
  --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --volume "$WS":/home/dev/ugv_ws \
  "$IMAGE" \
  bash