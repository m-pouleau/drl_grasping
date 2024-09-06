#!/bin/bash
cd /root
source /root/entrypoint.bash

nvidia-smi

#ros2 run drl_grasping ex_random_agent.bash