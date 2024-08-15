import abc
from collections import deque
from typing import Tuple

import gym
import numpy as np
from gym_ignition.utils.typing import Observation, ObservationSpace

from drl_grasping.envs.models.sensors import Camera
from drl_grasping.envs.perception import CameraSubscriber, PointCloudCreator
from drl_grasping.envs.tasks.reach import Reach


class ReachPointCloud(Reach, abc.ABC):

    _pointcloud_min_bound: Tuple[float, float, float] = (0.15, -0.3, 0.0)
    _pointcloud_max_bound: Tuple[float, float, float] = (0.75, 0.3, 0.6)

    def __init__(
        self,
        pointcloud_reference_frame_id: str,
        pointcloud_min_bound: Tuple[float, float, float],
        pointcloud_max_bound: Tuple[float, float, float],
        pointcloud_include_color: bool,
        pointcloud_include_intensity: bool,
        pointcloud_n_stacked: int,
        num_points: int = 2048,
        camera_type: str = "rgbd_camera",
        **kwargs,
    ):

        # Initialize the Task base class
        Reach.__init__(
            self,
            **kwargs,
        )

        # Store parameters for later use
        self._pointcloud_n_stacked = pointcloud_n_stacked
        self._num_points = num_points
        # Define number of channels depending on color features
        if pointcloud_include_color:
            self._num_pc_channels = 9
        elif pointcloud_include_intensity:
            self._num_pc_channels = 7
        else:
            self._num_pc_channels = 6

        # Perception (RGB-D camera - point cloud)
        self.camera_sub = CameraSubscriber(
            node=self,
            topic=Camera.get_points_topic(camera_type),
            is_point_cloud=True,
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
        self.pointcloud_creator = PointCloudCreator(
            node=self,
            tf2_listener=self.tf2_listener,
            reference_frame_id=self.substitute_special_frame(pointcloud_reference_frame_id),
            min_bound=pointcloud_min_bound,
            max_bound=pointcloud_max_bound,
            include_color=pointcloud_include_color,
            include_intensity=pointcloud_include_intensity,
            num_points = num_points
        )

        # Variable initialisation
        self.__stacked_pointclouds = deque([], maxlen=self._pointcloud_n_stacked)

    def create_observation_space(self) -> ObservationSpace:
        # Dimension num_points x num_channels
        return gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._pointcloud_n_stacked, self._num_points, self._num_pc_channels),
            dtype=np.float32,
        )

    def get_observation(self) -> Observation:

        # Get the latest point cloud
        point_cloud = self.camera_sub.get_observation()

        # Get point cloud in right format
        np_pointcloud = self.pointcloud_creator(point_cloud)

        self.__stacked_pointclouds.append(np_pointcloud)
        # For the first buffer after reset, fill with identical observations until deque is full
        while not self._pointcloud_n_stacked == len(self.__stacked_pointclouds):
            self.__stacked_pointclouds.append(np_pointcloud)

        # Create the observation
        observation = Observation(np.array(self.__stacked_pointclouds, dtype=np.float32))

        self.get_logger().debug(f"\nobservation: {observation}")

        # Return the observation
        return observation

    def reset_task(self):

        self.__stacked_pointclouds.clear()
        Reach.reset_task(self)
