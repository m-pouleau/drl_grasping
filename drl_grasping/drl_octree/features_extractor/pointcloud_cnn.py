import os
import gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from drl_grasping.drl_octree.features_extractor.pointnet import *
from drl_grasping.drl_octree.features_extractor.pointnet2 import *
from drl_grasping.drl_octree.features_extractor.diffusion_policy_3D import *
from drl_grasping.drl_octree.features_extractor.modules import LinearRelu


class PointCloudCnnFeaturesExtractor(BaseFeaturesExtractor):
    """
    :param observation_space:
    :param image_channels: Number of input image channels.
    :param normal_channels: Bool variable if normal channels are used or not
    :param channel_multiplier: Multiplier for the number of channels after each pooling.
                               With this parameter set to 1, the channels are [1, 2, 4, 8, ...] for [depth, depth-1, ..., full_depth].
    :param features_dim: Dimension of output feature vector. Note that this number is multiplied by the number of stacked inside one observation.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        normal_channels: bool = True, 
        image_channels: int = 3,
        features_dim: int = 248,
        aux_obs_dim: int = 10,
        aux_obs_features_dim: int = 8,
        extractor_backbone: str = "PointNet",
        verbose: bool = False,
    ):
        self._aux_obs_dim = aux_obs_dim
        self._aux_obs_features_dim = aux_obs_features_dim
        self._verbose = verbose

        # Determine number of stacked based on observation space shape
        self._n_stacks = observation_space.shape[0]

        # Chain up parent constructor
        super(PointCloudCnnFeaturesExtractor, self).__init__(
            observation_space, self._n_stacks * (features_dim + aux_obs_features_dim)
        )

        # Determine if feature extractor runs on gpu or cpu
        if torch.cuda.is_available():
            DEVICE = 'cuda'
        else:
            DEVICE = 'cpu'

        # Determine if normals are used by network (color input isn't considered, as ModelNet40 doesn't have color features)        
        if image_channels == 3:
            prefix = "seg_"
            num_channels = 9
        elif normal_channels:
            prefix = "normals_"
            num_channels = 6
        else:
            prefix = ""
            num_channels = 3

        # Get relative path of directory for pretrained models
        script_directory = os.path.dirname(os.path.abspath(__file__))

        extractor_backbone = "DP3"

        # Initialize the right feature extractor
        if extractor_backbone == "PointNet":
            weights_file_path = f"{script_directory}/weights/pointnet_{prefix}pretrained.pth"
            self._extractor_backbone = PointNetFeatureExtractor(num_channels=num_channels, features_dim=features_dim, file_path=weights_file_path, device=DEVICE)
        elif extractor_backbone == "PointNet2":
            weights_file_path = f"{script_directory}/weights/pointnet2_msg_{prefix}pretrained.pth"
            self._extractor_backbone = PointNet2FeatureExtractor(num_channels=num_channels, features_dim=features_dim, file_path=weights_file_path, device=DEVICE)
        elif extractor_backbone == "DP3":
            print("EXTRACTOR IS DP3 !!!", flush=True)
            self._extractor_backbone = DP3Extractor(color_channels=image_channels, num_channels=num_channels, features_dim=features_dim)

        # One linear layer for auxiliary observations
        if self._aux_obs_dim != 0:
            self.aux_obs_linear = LinearRelu(
                self._aux_obs_dim, aux_obs_features_dim
            )

        number_of_learnable_parameters = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        print(
            "Initialised PointCloudCnnFeaturesExtractor with "
            f"{number_of_learnable_parameters} parameters"
        )
        if verbose:
            print(self)

    def forward(self, obs):

        data = obs[0]
        print("Data is cuda: ", data.is_cuda, flush=True)
        aux_obs = obs[1]

        # Pass the data through the pointnet-based feature extractor backbone
        data = self._extractor_backbone(data)

        # Get a view that merges stacks into a single feature vector (original batches remain separated)
        data = data.view(-1, self._n_stacks * data.shape[-1])

        if self._aux_obs_dim != 0:
            # Feed the data through linear layer
            aux_data = self.aux_obs_linear(aux_obs.view(-1, self._aux_obs_dim))
            # Get a view that merges aux feature stacks into a single feature vector (original batches remain separated)
            aux_data = aux_data.view(
                -1, self._n_stacks * self._aux_obs_features_dim
            )
            # Concatenate auxiliary data
            data = torch.cat((data, aux_data), dim=1)

        return data


if __name__ == "__main__":
    import numpy as np
    
    # Set the device for the models
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'
    print("Device: ", DEVICE)    

    # Input Parameters
    COLORS_CHANNELS = 3
    USE_NORMALS = False
    if COLORS_CHANNELS == 3:
        NUM_CHANNELS = 12
    elif USE_NORMALS:
        NUM_CHANNELS = 9
    else:
        NUM_CHANNELS = 6
    NUM_FRAMES = 2
    NUM_POINTS = 1024
    PROPRIOCEPTIVE_OBSERVATIONS = True

    # Model Input
    pointcloud_input = torch.randn((NUM_FRAMES, NUM_POINTS, NUM_CHANNELS), dtype=torch.float32).to(DEVICE)
    print("Pointcloud: ", pointcloud_input.shape, "   CUDA: ", pointcloud_input.is_cuda)
    if PROPRIOCEPTIVE_OBSERVATIONS:
        AUX_DIM = 10
        aux_input = torch.randn((1, NUM_FRAMES, AUX_DIM), dtype=torch.float32).to(DEVICE)
        print("Auxiliary Input: ", aux_input.shape, "   CUDA: ", aux_input.is_cuda)
    else:
        AUX_DIM = 0
        aux_input = None

    # Model Observation Space
    mySpace = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2, NUM_POINTS+AUX_DIM, NUM_CHANNELS),dtype=np.float32)

    # Define Feature Extractor
    MyExtractor = PointCloudCnnFeaturesExtractor(observation_space=mySpace,
                                                 image_channels=COLORS_CHANNELS,
                                                 normal_channels=USE_NORMALS, 
                                                 features_dim=248,
                                                 aux_obs_dim=AUX_DIM,
                                                 aux_obs_features_dim=8,
                                                 extractor_backbone="PointNet",
                                                 ).to(DEVICE)

    # Data output
    data = MyExtractor((pointcloud_input, aux_input))
    print("Final Output: ", data.shape)
