#!/bin/bash
PROJECT="$(cd "$(dirname "$0")" && pwd)"
IMAGE="dasfl-gpu:v1"
echo "================================================"
echo "  DAS-FL GPU Experiment"
echo "  Args: $@"
echo "  Project: ${PROJECT}"
echo "  Time: $(date)"
echo "================================================"
docker run --gpus all --rm \
    -v ${PROJECT}:/workspace \
    -w /workspace \
    ${IMAGE} \
    python pipeline/run_fl.py "$@"
