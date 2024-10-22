from typing import List, Tuple

import numpy as np
import open3d
from rclpy.node import Node

from drl_grasping.envs.utils import Tf2Listener, conversions


class RGBDPointCloudCreator:
    def __init__(
        self,
        node: Node,
        tf2_listener: Tf2Listener,
        reference_frame_id: str,
        depth_max_distance: float = 5.0,
        img_height: int = 128,
        img_width: int = 128,
        camera_horizontal_fov: float = np.pi / 3.0,
        camera_vertical_fov: float = np.pi / 3.0,
        min_bound: Tuple[float, float, float] = (-1.0, -1.0, -1.0),
        max_bound: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        include_normals: bool = False,
        include_color: bool = False,
        # Note: For efficiency, the first channel of RGB is used for intensity
        include_intensity: bool = False,
        num_points: int = 1024,
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
        self._include_normals = include_normals
        self._include_color = include_color
        self._include_intensity = include_intensity
        self._num_points = num_points
        self._normals_radius = normals_radius
        self._normals_max_nn = normals_max_nn
        self._debug_draw = debug_draw
        self.get_workspace_center_and_radius()

        # Camera intrinsics (assumed values, replace with your actual values)
        self._depth_max_distance = depth_max_distance
        self._img_height = img_height
        self._img_width = img_width
        self._num_pixels = img_width * img_height
        self._fx = img_width / (2 * np.tan(camera_horizontal_fov / 2))  # focal length x
        self._fy = img_height / (2 * np.tan(camera_vertical_fov / 2))  # focal length y
        self._cx = img_width / 2  # principal point x
        self._cy = img_height / 2  # principal point y

        # Create a mesh grid of pixel coordinates
        x_indices, y_indices = np.meshgrid(np.arange(self._img_width), np.arange(self._img_height))
        self._x_flat, self._y_flat = x_indices.flatten(), y_indices.flatten()


    def __call__(self, depth_image_msg, color_image_msg) -> np.ndarray:

        # Get camera frame id
        camera_frame_id = depth_image_msg.header.frame_id

        # Convert to ndarray
        depth_image_flat = np.ndarray(
            buffer=depth_image_msg.data,
            dtype=np.float32,
            shape=(self._num_pixels,),
        )
        # Replace nan and inf with zero
        np.nan_to_num(depth_image_flat, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        if self._include_intensity or self._include_color:
            # Convert to ndarray
            color_image_flat = np.ndarray(
                    buffer=color_image_msg.data,
                    dtype=np.uint8,
                    shape=(self._num_pixels, 3),
                )
            if self._include_intensity:
                # Use only the first channel as the intensity observation
                color_image_flat = color_image_flat.reshape(
                    self._img_width, self._img_height, 3
                )[:, :, 0].reshape(-1)
            # Filter points based on clip_min and clip_max for depth
            valid_depth_mask = (depth_image_flat <= self._depth_max_distance) & (depth_image_flat > 0)
            # Filter points based on clip_min and clip_max for color
            valid_colors_mask = (color_image_flat[:, 0] >= 0) & (color_image_flat[:, 0] <= 255) & \
                                (color_image_flat[:, 1] >= 0) & (color_image_flat[:, 1] <= 255) & \
                                (color_image_flat[:, 2] >= 0) & (color_image_flat[:, 2] <= 255)
            # Combined mask
            valid_mask = valid_depth_mask & valid_colors_mask
            # Apply the mask to get valid points and colors
            X = depth_image_flat[valid_mask]
            x_2D = self._x_flat[valid_mask]
            y_2D = self._y_flat[valid_mask]
            color_image = color_image_flat[valid_mask]
            # Calculate 3D coordinates with broadcasting
            # -> Convention: X forward, Y left & Z up in image (not the classical camera coordinates)
            Y = -(x_2D - self._cx) * X / self._fx
            Z = -(y_2D - self._cy) * X / self._fy
            # Stack to create a (N, 3) array of points
            points = np.stack((X, Y, Z), axis=-1).astype(dtype=np.float32)
            # Normalize color
            colors = color_image.astype(dtype=np.float32) / 255.0

        else:
            # Filter points based on clip_min and clip_max for depth
            valid_depth_mask = (depth_image_flat <= self._depth_max_distance) & (depth_image_flat > 0)
            # Apply the mask to get valid points and colors
            X = depth_image_flat[valid_depth_mask]
            x_2D = self._x_flat[valid_depth_mask]
            y_2D = self._y_flat[valid_depth_mask]
            # Calculate 3D coordinates with broadcasting
            Y = -(x_2D - self._cx) * X / self._fx
            Z = -(y_2D - self._cy) * X / self._fy
            # Stack to create a (N, 3) array of points
            points = np.stack((X, Y, Z), axis=-1).astype(dtype=np.float32)
            colors = None

        # Preprocess point cloud (transform to robot frame, crop to workspace and estimate normals)
        open3d_point_cloud = self.preprocess_point_cloud(
            points,
            colors,
            camera_frame_id=camera_frame_id,
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

        # Adjust size of pointcloud to fitting number of points
        open3d_point_cloud = self.adjust_pointcloud_size(open3d_point_cloud, self._num_points, mode='random')

        # Convert open3d point cloud into numpy pointcloud & normalize xyz points to current workspace
        np_pointcloud = self.open3d_pointcloud_to_numpy_pointcloud(open3d_point_cloud=open3d_point_cloud)

        return np_pointcloud


    def preprocess_point_cloud(
        self,
        points: np.array,
        colors: np.array,
        camera_frame_id: str,
        reference_frame_id: str,
        min_bound: List[float],
        max_bound: List[float],
        normals_radius: float,
        normals_max_nn: int,
    ) -> open3d.geometry.PointCloud:
        
        # Create output Open3D PointCloud
        open3d_point_cloud = open3d.geometry.PointCloud()
        open3d_point_cloud.points = open3d.utility.Vector3dVector(points.astype(np.float64))
        if colors is not None:
            open3d_point_cloud.colors = open3d.utility.Vector3dVector(colors.astype(np.float64))

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
        if self._include_normals:
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
        # Get the point & normal features from the pointcloud, normalize xyz points to current workspace
        np_points = np.asarray(open3d_point_cloud.points)

        # Cases where the pretrained segmentation network is used as encoder
        # Get the color features & concatenate with other features
        if self._include_color:
            xyz, normed_xyz, xyz_skip = self.normalize_pointcloud_points(np_points)
            np_colors = np.asarray(open3d_point_cloud.colors)
            np_pointcloud = np.concatenate((xyz, np_colors, normed_xyz, xyz_skip), axis=1)
        elif self._include_intensity:
            xyz, normed_xyz, xyz_skip = self.normalize_pointcloud_points(np_points)
            np_colors = np.asarray(open3d_point_cloud.colors)[:, 0].reshape(-1, 1)
            np_pointcloud = np.concatenate((xyz, np_colors, normed_xyz, xyz_skip), axis=1)
        # Cases where the pretrained classifier is used as encoder
        else:
            xyz_norm, xyz_skip = self.pc_normalize(np_points)
            if self._include_normals:
                np_normals = np.asarray(open3d_point_cloud.normals)
                np_pointcloud = np.concatenate((xyz_norm, np_normals, xyz_skip), axis=1)
            else:
                np_pointcloud = np.concatenate((xyz_norm, xyz_skip), axis=1)

        return np_pointcloud


    def adjust_pointcloud_size(self, open3d_point_cloud, desired_num_points=1024, mode='random'):
        '''
        Sample pointcloud, so that it fits the specified number of points
        Modes: farthest point sampling & random sampling
        '''
        # Check if pointcloud needs to be sampled or padded
        downsample_ratio = desired_num_points / len(open3d_point_cloud.points)
        if downsample_ratio < 1:
            # use farthest point sampling to reduce size
            if mode == 'fps':
                open3d_point_cloud = open3d_point_cloud.farthest_point_down_sample(desired_num_points)
            # use random sampling to reduce size
            elif mode == 'random':
                # fix rounding error so number of points always ends up exactly desired_num_points
                expected_num_points = int(downsample_ratio * len(open3d_point_cloud.points))
                if expected_num_points < desired_num_points:
                    downsample_ratio = (desired_num_points + 0.5) / len(open3d_point_cloud.points)
                open3d_point_cloud = open3d_point_cloud.random_down_sample(downsample_ratio)
        # Padd pointcloud with zero entries if pointcloud is too small
        elif downsample_ratio > 1:
            padding_points = np.zeros((desired_num_points - len(open3d_point_cloud.points), 3))
            updated_points = np.vstack((np.asarray(open3d_point_cloud.points), padding_points))
            open3d_point_cloud.points = open3d.utility.Vector3dVector(updated_points)
        
        return open3d_point_cloud


    def get_workspace_center_and_radius(self):
        '''
        Get center and radius of workspace from task to be able to normalize the pointcloud later
        '''
        self._ws_center = (np.array(self._min_bound) + np.array(self._max_bound)) / 2
        self._ws_radius = np.linalg.norm(self._max_bound - self._ws_center)
        self._adjusted_ws_max_bound = np.array(self._max_bound) - np.array(self._min_bound)


    def pc_normalize(self, xyz_points):
        '''
        Normalize xyz points of pointcloud, so that they are fitted to workspace center and radius
        '''
        xyz_points -= self._ws_center
        xyz_norm = xyz_points.copy() / self._ws_radius
        xyz_skip = xyz_points.copy() / (self._adjusted_ws_max_bound * 0.5)
        return xyz_norm, xyz_skip

    def normalize_pointcloud_points(self, points):
        '''
        Normalize xyz points of pointcloud, so that they are fitting to pretrained segmentation model
        '''        
        xyz_points = points.copy()
        xyz_points[:, 0] -= self._ws_center[0]
        xyz_points[:, 1] -= self._ws_center[1]

        xyz_skip = xyz_points.copy()
        xyz_skip[:, 2] -= self._ws_center[2]
        xyz_skip = xyz_skip / (self._adjusted_ws_max_bound * 0.5)

        xyz_normalized = (points - np.array(self._min_bound)) / self._adjusted_ws_max_bound

        return xyz_points, xyz_normalized, xyz_skip
