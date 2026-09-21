#!/bin/bash
# Launch the Go-RIO container.
#
# DATA_DIR is mounted at /root/data and must point at the drive or folder that
# CONTAINS the dataset, not at the dataset itself: the launch files reference
# bags as /root/data/4dradarslam/NTU4Dradlm/<seq>/<seq>.bag. Set it to wherever
# your copy lives:
#   DATA_DIR=/path/to/your/data ./run.sh
#
# HOST_SHARED_DIR is the catkin workspace, mounted at /root/catkin_ws. It
# defaults to the workspace containing this repository, and it is also where the
# nodelet writes its trajectory output, so results land on the host directly.

set -u

export HOST_SHARED_DIR="${HOST_SHARED_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
export DATA_DIR="${DATA_DIR:-/media/$USER/ws_ssdP}"
export IMAGE="${IMAGE:-wooseong0929/go-rio:latest}"

if [ ! -d "$DATA_DIR" ]; then
    echo "error: DATA_DIR does not exist: $DATA_DIR" >&2
    exit 1
fi

xhost + >/dev/null

# glibc gives each thread its own malloc arena and does not hand the freed
# memory back, so a multi-threaded nodelet allocating a point cloud per frame
# on a 20-core host can hold several gigabytes it is no longer using: an nyl
# run sat at 7.3 GB resident with only 1.46 GB of it on the heap. Two arenas
# is plenty for this workload.
docker run --gpus all --rm -it --ipc=host --net=host --privileged \
    -e MALLOC_ARENA_MAX=2 \
    --env="DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$HOST_SHARED_DIR":/root/catkin_ws \
    -v "$DATA_DIR":/root/data \
    "$IMAGE" \
    /bin/bash

xhost - >/dev/null
