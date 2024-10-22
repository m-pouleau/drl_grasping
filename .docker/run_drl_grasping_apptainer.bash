#!/usr/bin/env bash
pwd; hostname; date

echo "Assigned GPUs: $SLURM_JOB_GPUS"

TAG="drl_grasping_apptainer.sif"

# Create all mounted repositories necessary for running container
mkdir -p /tmp/.ros/log
mkdir -p /tmp/.pcg
mkdir -p /tmp/.rviz2
mkdir -p /tmp/.config/matplotlib
mkdir -p /tmp/.cache/matplotlib
mkdir -p /tmp/.ignition/gazebo
mkdir -p /tmp/.ignition/log
chmod 777 /tmp/.ros/log /tmp/.pcg /tmp/.rviz2 /tmp/.config/matplotlib /tmp/.cache/matplotlib /tmp/.ignition/gazebo /tmp/.ignition/log

# Set to 1 if GPU's should be used 
export USE_SINGULARITY_NV=1

## Forward custom volumes variables
CUSTOM_VOLUMES=()
# Synchronize timezone with host
CUSTOM_VOLUMES+=("/etc/localtime:/etc/localtime:ro")
# Persistent storage of logs
CUSTOM_VOLUMES+=("$(dirname "${PWD}")/drl_grasping_training_docker:/root/drl_grasping_training")
# Add directories from mpouleau/drl_grasping
CUSTOM_VOLUMES+=("/tmp/.ignition:/root/.ignition")
CUSTOM_VOLUMES+=("/tmp/.pcg:/root/.pcg")
CUSTOM_VOLUMES+=("/tmp/.ros:/root/.ros")
CUSTOM_VOLUMES+=("/tmp/.rviz2:/root/.rviz2")
CUSTOM_VOLUMES+=("/tmp/.config/matplotlib:/root/.config/matplotlib")
CUSTOM_VOLUMES+=("/tmp/.cache/matplotlib:/root/.cache/matplotlib")
CUSTOM_VOLUMES+=("${PWD}/visualizations:/root/visualizations")
CUSTOM_VOLUMES+=("${PWD}/scripts:/root/ws/src/drl_grasping/scripts")
CUSTOM_VOLUMES+=("${PWD}/launch:/root/ws/src/drl_grasping/launch")
CUSTOM_VOLUMES+=("${PWD}/examples:/root/ws/src/drl_grasping/examples")
CUSTOM_VOLUMES+=("${PWD}/hyperparams:/root/ws/src/drl_grasping/hyperparams")
CUSTOM_VOLUMES+=("${PWD}/rviz:/root/ws/src/drl_grasping/rviz")
CUSTOM_VOLUMES+=("${PWD}/drl_grasping:/root/ws/src/drl_grasping/drl_grasping")

## Forward custom environment variables
CUSTOM_ENVS=()
# Synchronize ROS_DOMAIN_ID with host
if [ -n "${ROS_DOMAIN_ID}" ]; then
    CUSTOM_ENVS+=("ROS_DOMAIN_ID=${ROS_DOMAIN_ID}")
fi
# Synchronize IGN_PARTITION with host
if [ -n "${IGN_PARTITION}" ]; then
    CUSTOM_ENVS+=("IGN_PARTITION=${IGN_PARTITION}")
fi

## Fix Nvidia GPU Driver usage
nv=""
if [ "$USE_SINGULARITY_NV" = "1" ]; then
	# When running on Ubuntu newer than 20.04 host system, the host glibc is newer than 2.31 and this causes problems
	# together with the --nv switch because Singularity mounts too much files from the host inside the container.
	# So, if running on a newer host, we need to build a subset of the list of mounted libraries and exclude those that
	# are not directly related to the nvidia driver.
	glibc_version="$(ldd --version | grep -i glibc | head -n1 | grep -o '2\.[0-9]\{2,\}' | head -n1 | cut -d. -f2)"
	if [[ "$glibc_version" -le 31 ]]; then
		nv="--nv"
	else
		echo "Your system uses glibc newer than 2.31 which is used inside the container. Implementing --nv switch via a workaround. Let's hope it works!"
		libs="$(singularity --debug exec --nv "$IMAGE" echo test 2>&1 | grep 'addLibsMount.*/lib/' | sed 's#^.*\(/lib/[^ ]*\) .*$#\1#')"
		libs="$(echo "$libs" | grep -v -e libEGL.so -e libGL.so -e libGLdispatch.so -e libGLX.so -e libOpenCL.so -e libOpenGL.so)"
		nv="$(echo "$libs" | sed 's/^/-B /g' | tr "\n" " ")"
		nv="$nv -B /usr/share/glvnd/egl_vendor.d/10_nvidia.json"
	fi
fi

DOCKER_RUN_CMD=(
    apptainer exec
    $nv
    "${CUSTOM_VOLUMES[@]/#/"--bind "}"
    "${CUSTOM_ENVS[@]/#/"--env "}"
    "$(dirname "${PWD}")/${TAG}"
    ./.docker/apptainer_commands.bash
)

echo -e "\033[1;30m${DOCKER_RUN_CMD[*]}\033[0m" | xargs

# shellcheck disable=SC2048
exec ${DOCKER_RUN_CMD[*]}
