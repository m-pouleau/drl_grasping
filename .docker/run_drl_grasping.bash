#!/usr/bin/env bash

TAG="andrejorsula/drl_grasping:latest"

## Forward custom volumes and environment variables
CUSTOM_VOLUMES=()
CUSTOM_ENVS=()
while getopts ":v:e:" opt; do
    case "${opt}" in
        v) CUSTOM_VOLUMES+=("${OPTARG}") ;;
        e) CUSTOM_ENVS+=("${OPTARG}") ;;
        *)
            echo >&2 "Usage: ${0} [-v VOLUME] [-e ENV] [TAG] [CMD]"
            exit 2
            ;;
    esac
done
shift "$((OPTIND - 1))"

## GPU
# Enable GPU either via NVIDIA Container Toolkit or NVIDIA Docker (depending on Docker version)
if dpkg --compare-versions "$(docker version --format '{{.Server.Version}}')" gt "19.3"; then
    GPU_OPT="--gpus all"
else
    GPU_OPT="--runtime nvidia"
fi

## GUI
# To enable GUI, make sure processes in the container can connect to the x server
XAUTH=/tmp/.docker.xauth

# Always delete /tmp/.docker.xauth (regardless of type) and recreate it
if [ -e ${XAUTH} ]; then
    rm -rf ${XAUTH}
fi

touch ${XAUTH}
chmod a+r ${XAUTH}

XAUTH_LIST=$(xauth nlist "${DISPLAY}")
if [ -n "${XAUTH_LIST}" ]; then
    # shellcheck disable=SC2001
    XAUTH_LIST=$(sed -e 's/^..../ffff/' <<<"${XAUTH_LIST}")
    echo "${XAUTH_LIST}" | xauth -f ${XAUTH} nmerge -
fi

# GUI-enabling volumes
GUI_VOLUMES=(
    "${XAUTH}:${XAUTH}"
    "/tmp/.X11-unix:/tmp/.X11-unix"
    "/dev/input:/dev/input"
)
# GUI-enabling environment variables
GUI_ENVS=(
    XAUTHORITY="${XAUTH}"
    QT_X11_NO_MITSHM=1
    DISPLAY="${DISPLAY}"
)

## Additional volumes
# Synchronize timezone with host
CUSTOM_VOLUMES+=("/etc/localtime:/etc/localtime:ro")
# Persistent storage of logs
CUSTOM_VOLUMES+=("$(dirname "${PWD}")/drl_grasping_training_docker:/root/drl_grasping_training")
# Add directories from AndrejOrsula/drl_grasping
CUSTOM_VOLUMES+=("${PWD}/visualizations:/root/visualizations")
CUSTOM_VOLUMES+=("${PWD}/scripts:/root/ws/src/drl_grasping/scripts")
CUSTOM_VOLUMES+=("${PWD}/launch:/root/ws/src/drl_grasping/launch")
CUSTOM_VOLUMES+=("${PWD}/examples:/root/ws/src/drl_grasping/examples")
CUSTOM_VOLUMES+=("${PWD}/hyperparams:/root/ws/src/drl_grasping/hyperparams")
CUSTOM_VOLUMES+=("${PWD}/rviz:/root/ws/src/drl_grasping/rviz")
CUSTOM_VOLUMES+=("${PWD}/drl_grasping:/root/ws/src/drl_grasping/drl_grasping")

## Additional environment variables
# Synchronize ROS_DOMAIN_ID with host
if [ -n "${ROS_DOMAIN_ID}" ]; then
    CUSTOM_ENVS+=("ROS_DOMAIN_ID=${ROS_DOMAIN_ID}")
fi
# Synchronize IGN_PARTITION with host
if [ -n "${IGN_PARTITION}" ]; then
    CUSTOM_ENVS+=("IGN_PARTITION=${IGN_PARTITION}")
fi

DOCKER_RUN_CMD=(
    docker run
    --interactive
    --tty
    --rm
    --network host
    --ipc host
    --privileged
    --security-opt "seccomp=unconfined"
    "${GUI_VOLUMES[@]/#/"--volume "}"
    "${GUI_ENVS[@]/#/"--env "}"
    "${GPU_OPT}"
    "${CUSTOM_VOLUMES[@]/#/"--volume "}"
    "${CUSTOM_ENVS[@]/#/"--env "}"
    "${TAG}"
)

echo -e "\033[1;30m${DOCKER_RUN_CMD[*]}\033[0m" | xargs

# shellcheck disable=SC2048
exec ${DOCKER_RUN_CMD[*]}
