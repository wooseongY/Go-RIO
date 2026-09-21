#!/bin/bash
# Build the Go-RIO image from docker/Dockerfile.
#
# The recipe starts from osrf/ros:noetic-desktop-full and compiles ceres and g2o
# from source, so the first build takes roughly 40 minutes. Nothing is pulled
# from a prebuilt image, so the result depends only on this repository.
#
# Override the tag if you keep several builds around:
#   IMAGE=go-rio:mine ./build.sh

set -eu

cd "$(dirname "$0")"
IMAGE="${IMAGE:-wooseong0929/go-rio:latest}"

docker build --force-rm -f Dockerfile -t "$IMAGE" .
echo "built $IMAGE"
