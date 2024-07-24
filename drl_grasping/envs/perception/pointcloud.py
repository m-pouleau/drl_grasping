from typing import List, Tuple

import numpy as np
import open3d
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2

from drl_grasping.envs.utils import Tf2Listener, conversions


class PointCloudCreator:
    def __init__(
        self,
        node: Node,
        tf2_listener: Tf2Listener,
        reference_frame_id: str,
        min_bound: Tuple[float, float, float] = (-1.0, -1.0, -1.0),
        max_bound: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        include_color: bool = False,
        # Note: For efficiency, the first channel of RGB is used for intensity
        include_intensity: bool = False,
        num_points: int = 2048,
        normals_radius: float = 0.05,
        normals_max_nn: int = 10,
        debug_draw: bool = False,
    ):

        self._node = node

        # Listener of tf2 transforms is shared with the owner
        self.__tf2_listener = tf2_listener

        # Parameters
        self._reference_frame_id = reference_frame_id
        self._min_bound = min_bound
        self._max_bound = max_bound
        self._include_color = include_color
        self._include_intensity = include_intensity
        self._num_points = num_points
        self._normals_radius = normals_radius
        self._normals_max_nn = normals_max_nn
        self._debug_draw = debug_draw

    def __call__(self, ros_point_cloud2: PointCloud2) -> np.ndarray:

        # Convert to Open3D PointCloud
        open3d_point_cloud = conversions.pointcloud2_to_open3d(
            ros_point_cloud2=ros_point_cloud2,
            include_color=self._include_color,
            include_intensity=self._include_intensity,
        )

        # Preprocess point cloud (transform to robot frame, crop to workspace and estimate normals)
        open3d_point_cloud = self.preprocess_point_cloud(
            open3d_point_cloud=open3d_point_cloud,
            camera_frame_id=ros_point_cloud2.header.frame_id,
            reference_frame_id=self._reference_frame_id,
            min_bound=self._min_bound,
            max_bound=self._max_bound,
            normals_radius=self._normals_radius,
            normals_max_nn=self._normals_max_nn,
        )

        # Draw if needed
        if self._debug_draw:
            open3d.visualization.draw_geometries(
                [
                    open3d_point_cloud,
                    open3d.geometry.TriangleMesh.create_coordinate_frame(
                        size=0.2, origin=[0.0, 0.0, 0.0]
                    ),
                ],
                point_show_normal=True,
            )

        # Convert open3d point cloud into octree points
        np_pointcloud = self.open3d_pointcloud_to_numpy_pointcloud(open3d_point_cloud=open3d_point_cloud)

        # Sample to fitting number of points
        np_pointcloud = self.adjust_pointcloud_size(np_pointcloud, self._num_points)

        return np_pointcloud

    def preprocess_point_cloud(
        self,
        open3d_point_cloud: open3d.geometry.PointCloud,
        camera_frame_id: str,
        reference_frame_id: str,
        min_bound: List[float],
        max_bound: List[float],
        normals_radius: float,
        normals_max_nn: int,
    ) -> open3d.geometry.PointCloud:

        # Check if point cloud has any points
        if not open3d_point_cloud.has_points():
            self._node.get_logger().warn(
                "Point cloud has no points. Pre-processing skipped."
            )
            return open3d_point_cloud

        # Get transformation from camera to robot and use it to transform point
        # cloud into robot's base coordinate frame
        if camera_frame_id != reference_frame_id:
            transform = self.__tf2_listener.lookup_transform_sync(
                target_frame=reference_frame_id, source_frame=camera_frame_id
            )
            transform_mat = conversions.transform_to_matrix(transform=transform)
            open3d_point_cloud = open3d_point_cloud.transform(transform_mat)

        # Crop point cloud to include only the workspace
        open3d_point_cloud = open3d_point_cloud.crop(
            bounding_box=open3d.geometry.AxisAlignedBoundingBox(
                min_bound=min_bound, max_bound=max_bound
            )
        )

        # Check if any points remain in the area after cropping
        if not open3d_point_cloud.has_points():
            self._node.get_logger().warn(
                "Point cloud has no points after cropping it to the workspace volume."
            )
            return open3d_point_cloud

        # Estimate normal vector for each cloud point and orient these towards the camera
        open3d_point_cloud.estimate_normals(
            search_param=open3d.geometry.KDTreeSearchParamHybrid(
                radius=normals_radius, max_nn=normals_max_nn
            ),
            fast_normal_computation=True,
        )

        open3d_point_cloud.orient_normals_towards_camera_location(
            camera_location=transform_mat[0:3, 3]
        )

        return open3d_point_cloud
    
    def open3d_pointcloud_to_numpy_pointcloud(self, open3d_point_cloud: open3d.geometry.PointCloud) -> np.ndarray:
        '''
        Gets an open3d pointcloud and creates a numpy array out of the points, normals and color featues
        The dimension of the tensor can be:
            - n x 6 (no color information)
            - n x 7 (only intensity value)
            - n x 9 (rgb value)
        '''
        # Get the point & normal features from the pointcloud
        np_points = np.asarray(open3d_point_cloud.points)
        np_normals = np.asarray(open3d_point_cloud.normals)

        # Get the color features (if available) & concatenate with other features
        if self._include_color:
            np_colors = np.asarray(open3d_point_cloud.colors)
            np_pointcloud = np.concatenate((np_points, np_normals, np_colors), axis=1)
        elif self._include_intensity:
            np_colors = np.asarray(open3d_point_cloud.colors)[:, 0].reshape(-1, 1)
            np_pointcloud = np.concatenate((np_points, np_normals, np_colors), axis=1)
        else:
            np_pointcloud = np.concatenate((np_points, np_normals), axis=1)

        return np_pointcloud

    def adjust_pointcloud_size(self, pointcloud, desired_num_points=2048):
        current_num_points, features = pointcloud.shape
        
        if current_num_points > desired_num_points:
            # Randomly sample 2048 rows without replacement
            sampled_indices = np.random.choice(current_num_points, desired_num_points, replace=False)
            pointcloud = pointcloud[sampled_indices]
        elif current_num_points < desired_num_points:
            # Pad with zeros to make up the difference
            padding = np.zeros((desired_num_points - current_num_points, features))
            pointcloud = np.vstack((pointcloud, padding))
        
        return pointcloud
