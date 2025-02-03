# Deep Reinforcement Learning for Robotic Grasping in an Unstructured Environment

This project focuses on applying deep reinforcement learning to acquire a robust policy that allows robots to grasp diverse objects from compact 3D observations in the form of octrees or point clouds. It builds on the work of Orsula et al., from the  [following paper](https://arxiv.org/abs/2208.00818), and the [following repo](https://github.com/AndrejOrsula/drl_grasping).

<p align="center" float="middle">
  <a href="https://www.youtube.com/watch?v=1-cudiW4eaU">
    <img width="100.0%" src="https://github.com/AndrejOrsula/master_thesis/raw/media/media/webp/sim_panda.webp"/>
  </a>
  <em>Evaluation of a trained policy on novel scenes (previously unseen camera poses, objects, terrain textures, ...).</em>
</p>

<p align="center" float="middle">
  <a href="https://www.youtube.com/watch?v=btxqzFOgCyQ">
    <img width="100.0%" src="https://github.com/AndrejOrsula/master_thesis/raw/media/media/webp/sim2real.webp"/>
  </a>
  <em>Sim-to-Real transfer of a policy trained solely inside a simulation (zero-shot transfer). Credit: Aalborg University</em>
</p>

<p align="center" float="middle">
  <a href="https://www.youtube.com/watch?v=FZSoOkK6VFc">
    <img width="100.0%" src="https://github.com/AndrejOrsula/master_thesis/raw/media/media/webp/sim_lunar.webp"/>
  </a>
  <em>Evaluation of a trained policy for grasping rocks on the Moon inside a simulation.</em>
</p>

<p align="center" float="middle">
  <a href="https://www.youtube.com/watch?v=FZSoOkK6VFc">
    <img width="100.0%" src="https://github.com/AndrejOrsula/master_thesis/raw/media/media/webp/sim2real_lunar.webp"/>
  </a>
  <em>Sim-to-Real transfer in a Moon-analogue facility (zero-shot transfer). Credit: University of Luxembourg</em>
</p>

## Overview

<p align="center">
  <a href="https://docs.ros.org/en/galactic">
    <img src="https://img.shields.io/badge/Middleware-ROS%202%20Galactic-38469E"/>
  </a>
  <a href="https://gazebosim.org">
    <img src="https://img.shields.io/badge/Robotics%20Simulator-Gazebo%20Fortress-F58113"/>
  </a>
  <a href="https://moveit.ros.org">
    <img src="https://img.shields.io/badge/Motion%20Planning-MoveIt%202-0A58F7"/>
  </a>
  <br>
  <a href="https://www.gymlibrary.ml">
    <img src="https://img.shields.io/badge/RL%20Environment%20API-OpenAI%20Gym-CBCBCC"/>
  </a>
  <a href="https://stable-baselines3.readthedocs.io">
    <img src="https://img.shields.io/badge/Primary%20RL%20Framework-Stable--Baselines3-BDF25E"/>
  </a>
</p>

This repository contains multiple RL environments for robotic manipulation, focusing on robotic grasping using continuous actions in Cartesian space. All environments have several observation variants that enable direct comparison (RGB images, depth maps, octrees, point-clouds, ...). Each task is coupled with a simulation environment that can be used to train RL agents. These agents can subsequently be evaluated on real robots that integrate [ros2_control](https://control.ros.org) (or [ros_control](https://wiki.ros.org/ros_control) via [ros1_bridge](https://github.com/ros2/ros1_bridge)).

End-to-end model-free actor-critic algorithms have been tested on these environments ([TD3](https://arxiv.org/abs/1802.09477), [SAC](https://arxiv.org/abs/1801.01290) and [TQC](https://arxiv.org/abs/2005.04269) | [SB3 PyTorch implementation](https://github.com/DLR-RM/stable-baselines3)).

<details open><summary><b>List of Environments</b></summary>

Below is the list of implemented environments. Each environment (observation variant) has two alternatives, `Task-Obs-vX` and `Task-Obs-Gazebo-vX` (omitted from the table). Here, `Task-Obs-vX` implements the logic of the environment and can be used on real robots, whereas `Task-Obs-Gazebo-vX` combines this logic with the simulation environment inside Gazebo. Robots should be interchangeable for most parts, with some limitations (e.g. `GraspPlanetary` task requires a mobile manipulator to randomize the environment fully).

If you are interested in configuring these environments, first take a look at the list of their parameters inside [Gym registration](./drl_grasping/envs/__init__.py) and then at their individual source code.

<div align="center" class="tg-wrap">
<table>
<thead>
  <tr align="center" valign="bottom">
    <th>
      <a href="./drl_grasping/envs/tasks/reach">
        <img width="100.0%" src="https://user-images.githubusercontent.com/22929099/177349186-978fa919-c2ab-40f2-b667-830c42c83ce8.png"/>
      </a>
      <em>Reach the end-effector goal.</em>
    </th>
    <th>
      <a href="./drl_grasping/envs/tasks/grasp">
        <img width="100.0%" src="https://user-images.githubusercontent.com/22929099/177349182-09a0202f-37b1-4240-82c1-3c00e5c17293.png"/>
      </a>
      <em>Grasp and lift a random object.</em>
    </th>
    <th>
      <a href="./drl_grasping/envs/tasks/grasp_planetary">
        <img width="100.0%" src="https://user-images.githubusercontent.com/22929099/177349185-037a1ed6-f46a-44e1-bba2-e1557d1b894c.png"/>
      </a>
      <em>Grasp and lift a Moon rock.</em>
    </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Reach-v0 (state obs)</td>
    <td>Grasp-v0 (state obs)</td>
    <td>GraspPlanetary-v0 (state obs)</td>
  </tr>
  <tr>
    <td align="center">—</td>
    <td align="center">—</td>
    <td>GraspPlanetary-MonoImage-v0</td>
  </tr>
  <tr>
    <td>Reach-ColorImage-v0</td>
    <td align="center">—</td>
    <td>GraspPlanetary-ColorImage-v0</td>
  </tr>
  <tr>
    <td>Reach-DepthImage-v0</td>
    <td align="center">—</td>
    <td>GraspPlanetary-DepthImage-v0</td>
  </tr>
  <tr>
    <td align="center">—</td>
    <td align="center">—</td>
    <td>GraspPlanetary-DepthImageWithIntensity-v0</td>
  </tr>
  <tr>
    <td align="center">—</td>
    <td align="center">—</td>
    <td>GraspPlanetary-DepthImageWithColor-v0</td>
  </tr>
  <tr>
    <td>Reach-Octree-v0</td>
    <td>Grasp-Octree-v0</td>
    <td>GraspPlanetary-Octree-v0</td>
  </tr>
  <tr>
    <td>Reach-PointNet-v0</td>
    <td>Grasp-PointNet-v0</td>
    <td>GraspPlanetary-PointNet-v0</td>
  </tr>
  <tr>
    <td>Reach-PointNet2-v0</td>
    <td>Grasp-PointNet2-v0</td>
    <td>GraspPlanetary-PointNet2-v0</td>
  </tr>
  <tr>
    <td>Reach-DP3-v0</td>
    <td>Grasp-DP3-v0</td>
    <td>GraspPlanetary-DP3-v0</td>
  </tr>
  <tr>
    <td>Reach-OctreeWithIntensity-v0</td>
    <td>Grasp-OctreeWithIntensity-v0</td>
    <td>GraspPlanetary-OctreeWithIntensity-v0</td>
  </tr>
  <tr>
    <td>Reach-OctreeWithColor-v0</td>
    <td>Grasp-OctreeWithColor-v0</td>
    <td>GraspPlanetary-OctreeWithColor-v0</td>
  </tr>
  <tr>
    <td>Reach-PointNetWithColor-v0</td>
    <td>Grasp-PoinNetWithColor-v0</td>
    <td>GraspPlanetary-PointNetWithColor-v0</td>
  </tr>
  <tr>
    <td>Reach-RGBDPointNetWithColor-v0</td>
    <td>Grasp-RGBDPoinNetWithColor-v0</td>
    <td>GraspPlanetary-RGBDPointNetWithColor-v0</td>
  </tr>
  <tr>
    <td>Reach-PointNet2WithColor-v0</td>
    <td>Grasp-PointNet2WithColor-v0</td>
    <td>GraspPlanetary-PointNet2WithColor-v0</td>
  </tr>
  <tr>
    <td>Reach-DP3WithColor-v0</td>
    <td>Grasp-DP3WithColor-v0</td>
    <td>GraspPlanetary-DP3WithColor-v0</td>
  </tr>
  <tr>
    <td>Reach-RGBD-DP3WithColor-v0</td>
    <td>Grasp-RGBD-DP3WithColor-v0</td>
    <td>GraspPlanetary-RGBD-DP3WithColor-v0</td>
  </tr>
</tbody>
</table>
</div>

By default, `Grasp` and `GraspPlanetary` tasks utilize [`GraspCurriculum`](./drl_grasping/envs/tasks/curriculums/grasp.py) that shapes their reward function and environment difficulty.

</details>

<details><summary><b>Domain Randomization</b></summary>

To facilitate the sim-to-real transfer of trained agents, simulation environments introduce domain randomization with the aim of improving the generalization of learned policies. This randomization is accomplished via [`ManipulationGazeboEnvRandomizer`](./drl_grasping/envs/randomizers/manipulation.py) that populates the virtual world and enables randomizing of several properties at each reset of the environment. As this randomizer is configurable with numerous parameters, please take a look at the source code to see what environments you can create.

<p align="center" float="middle">
  <a href="./drl_grasping/envs/randomizers/manipulation.py">
    <img width="100.0%" src="https://user-images.githubusercontent.com/22929099/177401924-134095c6-2b30-4529-8f81-d3c9e4d9144b.png"/>
  </a>
  <em>Examples of domain randomization for the <code>Grasp</code> task.</em>
</p>

<p align="center" float="middle">
  <a href="./drl_grasping/envs/randomizers/manipulation.py">
    <img width="100.0%" src="https://user-images.githubusercontent.com/22929099/181464827-90ec191a-3166-42f3-862c-415eff56e490.png"/>
  </a>
  <em>Examples of domain randomization for the <code>GraspPlanetary</code> task.</em>
</p>

#### Model Datasets

Simulation environments in this repository can utilize datasets of any [SDF](http://sdformat.org) models, e.g. models from [Fuel](https://app.gazebosim.org). By default, the `Grasp` task uses [Google Scanned Objects collection](https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research) together with a set of PBR textures pointed to by `TEXTURE_DIRS` environment variable. On the contrary, the `GraspPlanetary` task employs custom models that are procedurally generated via [Blender](https://blender.org). However, this can be adjusted if desired.

All external models can be automatically configured and randomized in several ways via [`ModelCollectionRandomizer`](./drl_grasping/envs/models/utils/model_collection_randomizer.py) before their insertion into the world, e.g. optimization of collision geometry, estimation of (randomized) inertial properties and randomization of parameters such as geometry scale or surface friction. When processing large collections, model filtering can also be enabled based on several aspects, such as the complexity of the geometry or the existence of disconnected components. A few scripts for managing datasets can be found under [scripts/utils/](./scripts/utils/) directory.

</details>

<details><summary><b>End-to-End Learning from 3D Octree Observations</b></summary>

In the project of the base paper from Andrej Orsula, it was initially investigated how 3D visual observations can be leveraged to improve end-to-end learning of manipulation skills. Octrees were selected for this purpose due to their efficiently organized structure compared to other 3D representations.

To enable the extraction of abstract features from 3D octree observations, an octree-based 3D CNN is employed. The network module that accomplishes such feature extraction is implemented in the form of [`OctreeCnnFeaturesExtractor`](./drl_grasping/drl_octree/features_extractor/octree_cnn.py) (PyTorch). This features extractor is part of the `OctreeCnnPolicy` policy implemented for TD3, SAC and TQC algorithms. Internally, the feature extractor utilizes [O-CNN](https://github.com/microsoft/O-CNN) implementation to benefit from hardware acceleration on NVIDIA GPUs.

<p align="center" float="middle">
  <a href="./drl_grasping/drl_octree/features_extractor/octree_cnn.py">
    <img width="100.0%" src="https://user-images.githubusercontent.com/22929099/176558147-600646ce-ff9c-4660-8300-532acb6df0e4.svg"/>
  </a>
  <em>Illustration of the end-to-end actor-critic network architecture with octree-based 3D CNN feature extractor.</em>
</p>

</details>

<details><summary><b>End-to-End Learning from 3D Point-Cloud Observations</b></summary>
Another approach is to directly utilize the point-cloud data and use an point-cloud encoder network, such as the PointNet-based encoder and the DP3-based encoder (more details on this in the Master Thesis Document). The following two figure show the architector of both approaches:
<br>
<br>

<p align="center" float="middle">
  <a href="https://private-user-images.githubusercontent.com/150341612/409142018-a309908c-f421-45bc-aced-6d4138d317c6.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Mzg1OTI4MzksIm5iZiI6MTczODU5MjUzOSwicGF0aCI6Ii8xNTAzNDE2MTIvNDA5MTQyMDE4LWEzMDk5MDhjLWY0MjEtNDViYy1hY2VkLTZkNDEzOGQzMTdjNi5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwMjAzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDIwM1QxNDIyMTlaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1iY2UxNzE0MDdhMzA0MTc3MWFiYWY0NWQzMjFkODNjNGI5N2ZkZTUxOWQzNzk2ODBkYWZlYTdlNDIzMGU0YzYxJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.4q-HDpUMTfuXmW35KURlDHHhdq2YvupB2eyUcuKGffc">
    <img width="100.0%" src="https://private-user-images.githubusercontent.com/150341612/409142018-a309908c-f421-45bc-aced-6d4138d317c6.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Mzg1OTI4MzksIm5iZiI6MTczODU5MjUzOSwicGF0aCI6Ii8xNTAzNDE2MTIvNDA5MTQyMDE4LWEzMDk5MDhjLWY0MjEtNDViYy1hY2VkLTZkNDEzOGQzMTdjNi5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwMjAzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDIwM1QxNDIyMTlaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1iY2UxNzE0MDdhMzA0MTc3MWFiYWY0NWQzMjFkODNjNGI5N2ZkZTUxOWQzNzk2ODBkYWZlYTdlNDIzMGU0YzYxJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.4q-HDpUMTfuXmW35KURlDHHhdq2YvupB2eyUcuKGffc"/>
  </a>
  <em>Illustration of the end-to-end actor-critic network architecture with PointNet-based 3D feature extractor.</em>
</p>
<p align="center" float="middle">
  <a href="https://private-user-images.githubusercontent.com/150341612/409142054-6ff210ef-382b-4077-9461-ed2855f7ad21.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Mzg1OTI4MzksIm5iZiI6MTczODU5MjUzOSwicGF0aCI6Ii8xNTAzNDE2MTIvNDA5MTQyMDU0LTZmZjIxMGVmLTM4MmItNDA3Ny05NDYxLWVkMjg1NWY3YWQyMS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwMjAzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDIwM1QxNDIyMTlaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT05ZGE5NzdhMjdhOWI4MzRmOTBlMzcyNGM4MzliZDc5ZjI5OWE1YzgxZDFmYmEwNmZmNjI2NmE5NWIwNTBjZjQ5JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.hJvcg79QA9UbrctMUP2GYRt4slTk_wCYJB__ovQVRv8">
    <img width="100.0%" src="https://private-user-images.githubusercontent.com/150341612/409142054-6ff210ef-382b-4077-9461-ed2855f7ad21.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Mzg1OTI4MzksIm5iZiI6MTczODU5MjUzOSwicGF0aCI6Ii8xNTAzNDE2MTIvNDA5MTQyMDU0LTZmZjIxMGVmLTM4MmItNDA3Ny05NDYxLWVkMjg1NWY3YWQyMS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwMjAzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDIwM1QxNDIyMTlaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT05ZGE5NzdhMjdhOWI4MzRmOTBlMzcyNGM4MzliZDc5ZjI5OWE1YzgxZDFmYmEwNmZmNjI2NmE5NWIwNTBjZjQ5JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.hJvcg79QA9UbrctMUP2GYRt4slTk_wCYJB__ovQVRv8"/>
  </a>
  <em>Illustration of the end-to-end actor-critic network architecture with 3D-Diffusion-Policy-based feature extractor.</em>
</p>
  
</details>

<details><summary><b>Limitations</b></summary>

The known limitations of this repository are listed below for your convenience.

- **No parallel environments –** It is currently not possible to run multiple instances of the environment simultaneously.
- **Slow training –** The simulation environments are computationally complex (physics, rendering, underlying low-level control, ...). This significantly impacts the ability to train agents with time and computational constraints. The performance of some of these aspects can be improved at the cost of accuracy and realism (e.g. `physics_rate`/`step_size`).
- **Suboptimal hyperparameters –** Although a hyperparameter optimization framework was employed for some combinations of environments and algorithms, it is a prolonged process. This problem is exacerbated by the vast quantity of hyperparameters and their general brittleness. Therefore, the default hyperparameters provided in this repository might not be optimal.
- **Nondeterministic –** Experiments are not fully repeatable, and even the same seed of the pseudorandom generator can lead to different results. This is caused by several aspects, such as the nondeterministic nature of network-based communication and non-determinism in the underlying deep learning frameworks and hardware.

</details>

## Instructions

For the setup, it is recommended to use the pre-built Docker imagae, due to its simplicity, as well the fact that it comes with all the required datasets while enabling isolation of runs.

<details><summary><b>Docker Installation</b></summary>

### Hardware Requirements

- **CUDA GPU –** CUDA-enabled GPU is required for hardware-accelerated processing of octree observations. Everything else should also be functional on the CPU.

### Install Docker

First, ensure your system has a setup for using Docker with NVIDIA GPUs. You can follow [`install_docker_with_nvidia.bash`](./.docker/host/install_docker_with_nvidia.bash) installation script for Debian-based distributions. Alternatively, consult the [NVIDIA Container Toolkit Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for other Linux distributions.

```bash
# Execute script inside a cloned repository
.docker/host/install_docker_with_nvidia.bash
```

### Clone a Prebuilt Docker Image

Prebuilt Docker images of `drl_grasping` can be pulled directly from [Docker Hub](https://hub.docker.com/repository/docker/mpouleau/drl_grasping) without needing to build them locally. You can use the following command to manually pull the latest image or one of the previous tagged [Releases](https://github.com/m-pouleau/drl_grasping/releases). The average size of images is 29GB (including datasets).

```bash
docker pull mpouleau/drl_grasping:${TAG:-latest}
```

### (Optional) Build a New Image

It is also possible to build the Docker image locally using the included [Dockerfile](./Dockerfile). To do this, [`build.bash`](./.docker/build.bash) script can be executed as shown below (arguments are optional). This script will always print the corresponding low-level `docker build ...` command for your reference.

```bash
.docker/build.bash ${TAG:-latest} ${BUILD_ARGS}
```

### Run a Docker Container

For simplicity, please run `drl_grasping` Docker containers using the included [`run_drl_grasping.bash`](./.docker/run_drl_grasping.bash) script shown below (arguments are optional). It enables NVIDIA GPUs and GUI interface while automatically mounting the necessary volumes (e.g. persistent logging) and setting environment variables (e.g. synchronization of middleware communication with the host). This script will always print the corresponding low-level `docker run ...` command for your reference.

```bash
# Execute script inside a cloned repository
.docker/run_drl_grasping.bash ${TAG:-latest} ${CMD}
```

The network communication of `drl_grasping` within this Docker container is configured based on the ROS 2 [`ROS_DOMAIN_ID`](https://docs.ros.org/en/galactic/Concepts/About-Domain-ID.html) environment variable, which can be set via `ROS_DOMAIN_ID={0...101} .docker/run_drl_grasping.bash ${TAG:-latest} ${CMD}`. By default (`ROS_DOMAIN_ID=0`), external communication is restricted and multicast is disabled. With `ROS_DOMAIN_ID=42`, the communication remains restricted to `localhost` with multicast enabled, enabling monitoring of communication outside the container but within the same system. Using `ROS_DOMAIN_ID=69` will use the default network interface and multicast settings, which can enable monitoring of communication within the same LAN. All other `ROS_DOMAIN_ID`s share the default behaviour and can be employed to enable communication partitioning for running of multiple `drl_grasping` instances.

</details>

<details><summary><b>Test Random Agents</b></summary>

A good starting point is to simulate some episodes using random agents where actions are sampled from the defined action space. This is also useful when modifying environments because it lets you analyze the consequences of actions and resulting observations without deep learning pipelines running in the background. To get started, run the following example. It should open RViz 2 and Gazebo client instances that provide you with visual feedback.

```bash
ros2 run drl_grasping ex_random_agent.bash
```

After running the example script, the underlying `ros2 launch drl_grasping random_agent.launch.py ...` command with all arguments will always be printed for your reference (example shown below). If desired, you can launch this command directly with custom arguments.

```bash
ros2 launch drl_grasping random_agent.launch.py seed:=42 robot_model:=lunalab_summit_xl_gen env:=GraspPlanetary-Octree-Gazebo-v0 check_env:=false render:=true enable_rviz:=true log_level:=warn
```

</details>

<!-- <details><summary><b>[WIP] Try Pre-trained Agents</b></summary>

**Note:** Submodule `pretrained_agents` is currently incompatible with `drl_grasping` version `2.0.0`. Previously released versions using the Docker setup are functional if you want to test this feature.

Submodule [pretrained_agents](https://github.com/AndrejOrsula/drl_grasping_pretrained_agents) contains a selection of agents that are already trained and ready. To try them out, run the following example. It should open RViz 2 and Gazebo client instances that provide you with visual feedback, while the agent's performance will be logged and printed to `STDOUT`.

```bash
ros2 run drl_grasping ex_evaluate_pretrained_agent.bash
```

After running the example script, the underlying `ros2 launch drl_grasping evaluate.launch.py ...` command with all arguments will always be printed for your reference (example shown below). If desired, you can launch this command directly with custom arguments. For example, you can select what agent to try according to the support matrix from [AndrejOrsula/drl_grasping_pretrained_agents](./pretrained_agents/README.md).

```bash
ros2 launch drl_grasping evaluate.launch.py seed:=77 robot_model:=panda env:=Grasp-Octree-Gazebo-v0 algo:=tqc log_folder:=/root/ws/install/share/drl_grasping/pretrained_agents reward_log:=/root/drl_grasping_training/evaluate/Grasp-Octree-Gazebo-v0 stochastic:=false n_episodes:=200 load_best:=false enable_rviz:=true log_level:=error
```

</details> -->

<details><summary><b>Train New Agents</b></summary>

You can also train your agents from scratch. To begin the training, run the following example. By default, headless mode is used during the training to reduce computational load.

```bash
ros2 run drl_grasping ex_train.bash
```

After running the example script, the underlying `ros2 launch drl_grasping train.launch.py ...` command with all arguments will always be printed for your reference (example shown below). If desired, you can launch this command directly with custom arguments.

```bash
ros2 launch drl_grasping train.launch.py seed:=42 robot_model:=panda env:=Grasp-OctreeWithColor-Gazebo-v0 algo:=tqc log_folder:=/root/drl_grasping_training/train/Grasp-OctreeWithColor-Gazebo-v0/logs tensorboard_log:=/root/drl_grasping_training/train/Grasp-OctreeWithColor-Gazebo-v0/tensorboard_logs save_freq:=10000 save_replay_buffer:=true log_interval:=-1 eval_freq:=10000 eval_episodes:=20 enable_rviz:=false log_level:=fatal
```

#### TensorBoard

TensorBoard logs will be generated during training in a directory specified by the `tensorboard_log:=${TENSORBOARD_LOG}` argument. You can open them in your web browser using the following command.

```bash
tensorboard --logdir ${TENSORBOARD_LOG}
```

</details>

<details><summary><b>Evaluate New Agents</b></summary>

Once you train your agents, you can evaluate them. Start by looking at [ex_evaluate.bash](./examples/ex_evaluate.bash), which can be modified to fit your trained agent. It should open RViz 2 and Gazebo client instances that provide you with visual feedback, while the agent's performance will be logged and printed to `STDOUT`.

```bash
ros2 run drl_grasping ex_evaluate.bash
```

After running the example script, the underlying `ros2 launch drl_grasping evaluate.launch.py ...` command with all arguments will always be printed for your reference (example shown below). If desired, you can launch this command directly with custom arguments. For example, you can select a specific checkpoint with the `load_checkpoint:=${LOAD_CHECKPOINT}` argument instead of running the final model.

```bash
ros2 launch drl_grasping evaluate.launch.py seed:=77 robot_model:=panda env:=Grasp-Octree-Gazebo-v0 algo:=tqc log_folder:=/root/drl_grasping_training/train/Grasp-Octree-Gazebo-v0/logs reward_log:=/root/drl_grasping_training/evaluate/Grasp-Octree-Gazebo-v0 stochastic:=false n_episodes:=200 load_best:=false enable_rviz:=true log_level:=warn
```

</details>

## Directory Structure

```bash
.
├── drl_grasping/        # [dir] Primary Python module of this project
│   ├── drl_octree/      # [dir] Submodule for end-to-end learning from 3D octree observations
│   ├── envs/            # [dir] Submodule for environments
│   │   ├── control/     # [dir] Interfaces for the control of agents
│   │   ├── models/      # [dir] Functional models for simulation environments
│   │   ├── perception/  # [dir] Interfaces for the perception of agents
│   │   ├── randomizers/ # [dir] Domain randomization of the simulated environments
│   │   ├── runtimes/    # [dir] Runtime implementations of the task (sim/real)
│   │   ├── tasks/       # [dir] Implementation of tasks
│   │   ├── utils/       # [dir] Environment-specific utilities used across the submodule
│   │   └── worlds/      # [dir] Minimal templates of worlds for simulation environments
│   └── utils/           # [dir] Submodule for training and evaluation scripts boilerplate (using SB3)
├── examples/            # [dir] Examples for training and evaluating RL agents
├── hyperparams/         # [dir] Default hyperparameters for training RL agents
├── visualizations/      # [dir] Directory to access scripts to visualize tensorboard data
├── launch/              # [dir] ROS 2 launch scripts that can be used to interact with this repository
├── pretrained_agents/   # [dir] Collection of pre-trained agents
├── rviz/                # [dir] RViz2 config for visualization
├── scripts/             # [dir] Helpful scripts for training, evaluation and other utilities
├── CMakeLists.txt       # Colcon-enabled CMake recipe
└── package.xml          # ROS 2 package metadata
```
