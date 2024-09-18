#!/bin/bash
cd /root
source /root/entrypoint.bash

# Declare a variable to control output filtering
export FILTER_OUTPUT=true

ros2 run drl_grasping ex_train.bash