#!/bin/bash
cd /root
source /root/entrypoint.bash

# Set the environment variable for PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

ros2 run drl_grasping ex_train.bash