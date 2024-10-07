import abc
from collections import deque
from typing import Tuple

import gym
import numpy as np
from gym_ignition.utils.typing import Observation, ObservationSpace

from drl_grasping.envs.models.sensors import Camera
from drl_grasping.envs.perception import CameraSubscriber, RGBDPointCloudCreator
from drl_grasping.envs.tasks.grasp import Grasp
from drl_grasping.envs.utils.conversions import orientation_quat_to_6d


class GraspRGBDPointCloud(Grasp, abc.ABC):
    def __init__(
        self,
        pointcloud_reference_frame_id: str,
        pointcloud_min_bound: Tuple[float, float, float],
        pointcloud_max_bound: Tuple[float, float, float],
        pointcloud_include_normals: bool,
        pointcloud_include_color: bool,
        pointcloud_include_intensity: bool,
        pointcloud_n_stacked: int,
        proprioceptive_observations: bool,
        num_points: int = 1024,
        camera_type: str = "rgbd_camera",
        depth_max_distance: float = 5.0,
        camera_width: int = 128,
        camera_height: int = 128,
        camera_horizontal_fov: float = np.pi / 3.0,
        camera_vertical_fov: float = np.pi / 3.0,
        **kwargs,
    ):

        # Initialize the Task base class
        Grasp.__init__(
            self,
            **kwargs,
        )

        # Perception (depth map)
        self.camera_sub = CameraSubscriber(
            node=self,
            topic=Camera.get_depth_topic(camera_type),
            is_point_cloud=False,
            callback_group=self._callback_group,
        )
        # Perception (RGB image)
        if pointcloud_include_color or pointcloud_include_intensity:
            assert camera_type == "rgbd_camera"
            self.camera_sub_color = CameraSubscriber(
                node=self,
                topic=Camera.get_color_topic(camera_type),
                is_point_cloud=False,
                callback_group=self._callback_group,
            )

        # Offset pointcloud bounds by the robot base offset
        pointcloud_min_bound = (
            pointcloud_min_bound[0],
            pointcloud_min_bound[1],
            pointcloud_min_bound[2] + self.robot_model_class.BASE_LINK_Z_OFFSET,
        )
        pointcloud_max_bound = (
            pointcloud_max_bound[0],
            pointcloud_max_bound[1],
            pointcloud_max_bound[2] + self.robot_model_class.BASE_LINK_Z_OFFSET,
        )

        # Pointcloud creator
        self.pointcloud_creator = RGBDPointCloudCreator(
            node=self,
            tf2_listener=self.tf2_listener,
            reference_frame_id=self.substitute_special_frame(pointcloud_reference_frame_id),
            min_bound=pointcloud_min_bound,
            max_bound=pointcloud_max_bound,
            include_normals=pointcloud_include_normals,
            include_color=pointcloud_include_color,
            include_intensity=pointcloud_include_intensity,
            num_points = num_points,
            depth_max_distance = depth_max_distance,
            img_height = camera_width,
            img_width = camera_height,
            camera_horizontal_fov = camera_horizontal_fov,
            camera_vertical_fov = camera_vertical_fov
        )

        # Additional parameters
        self._proprioceptive_observations = proprioceptive_observations
        self._pointcloud_n_stacked = pointcloud_n_stacked
        self._num_points = num_points
        self._aux_dim = 10
        self._pointcloud_include_color = pointcloud_include_color
        self._pointcloud_include_intensity = pointcloud_include_intensity
        # Define number of channels depending on color & normal features
        self._num_pc_channels = 6
        if pointcloud_include_color:
            self._num_pc_channels += 6
        elif pointcloud_include_intensity:
            self._num_pc_channels += 4
        elif pointcloud_include_normals:
            self._num_pc_channels += 3

        # List of all stacked observations
        self.__stacked_observations = deque([], maxlen=self._pointcloud_n_stacked)


    def create_observation_space(self) -> ObservationSpace:
        # Space for proprioceptive state observations and pointcloud observations
        # 0   - (gripper) Gripper state
        #       - 1.0: opened
        #       - -1.0: closed
        # 1:4 - (x, y, z) displacement
        #       - metric units, unbound
        # 4:10 - (v1_x, v1_y, v1_z, v2_x, v2_y, v2_z) 3D orientation in "6D representation"
        #       - normalised
        
        if self._proprioceptive_observations:
            return gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self._pointcloud_n_stacked, self._num_points+self._aux_dim, self._num_pc_channels),
                dtype=np.float32,
            )
        else:
            return gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self._pointcloud_n_stacked, self._num_points, self._num_pc_channels),
                dtype=np.float32,
            )


    def get_observation(self) -> Observation:

        # Get the latest depth map
        depth_image_msg = self.camera_sub.get_observation()

        if self._pointcloud_include_color or self._pointcloud_include_intensity:
            # Get the latest color image
            color_image_msg = self.camera_sub_color.get_observation()

        assert depth_image_msg.height * depth_image_msg.width == len(depth_image_msg.data)
        assert (depth_image_msg.height != self._camera_width) and (depth_image_msg.width != self._camera_height)

        # Get point cloud in right format
        observation_array = self.pointcloud_creator(depth_image_msg, color_image_msg)

        # Gather proprioceptive observations if enabled
        if self._proprioceptive_observations:
            ee_position, ee_orientation = self.get_ee_pose()
            ee_orientation = orientation_quat_to_6d(quat_xyzw=ee_orientation)
            aux_obs = np.array((
                (1.0 if self.gripper.is_open else -1.0,)
                + ee_position
                + ee_orientation[0]
                + ee_orientation[1]
            ), dtype=np.float32).reshape(10, 1)
            observation_array = np.concatenate((observation_array, np.tile(aux_obs, (1, self._num_pc_channels))), axis=0)
        
        self.__stacked_observations.append(observation_array)
        # For the first buffer after reset, fill with identical observations until deque is full
        while not self._pointcloud_n_stacked == len(self.__stacked_observations):
            self.__stacked_observations.append(observation_array)

        observation = Observation(np.array(self.__stacked_observations, dtype=np.float32))

        self.get_logger().debug(f"\nobservation: {observation}")

        # Return the observation
        return observation


    def reset_task(self):

        self.__stacked_observations.clear()
        Grasp.reset_task(self)
