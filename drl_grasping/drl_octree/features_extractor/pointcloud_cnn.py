import gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from drl_grasping.drl_octree.features_extractor.pointnet import *
from drl_grasping.drl_octree.features_extractor.pointnet2 import *
from drl_grasping.drl_octree.features_extractor.modules import LinearRelu


class PointCloudCnnFeaturesExtractor(BaseFeaturesExtractor):
    """
    :param observation_space:
    :param channels_in: Number of input channels.
    :param channel_multiplier: Multiplier for the number of channels after each pooling.
                               With this parameter set to 1, the channels are [1, 2, 4, 8, ...] for [depth, depth-1, ..., full_depth].
    :param features_dim: Dimension of output feature vector. Note that this number is multiplied by the number of stacked inside one observation.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        channels_in: int = 7,
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

        # Determine if normals are used by network
        #TODO: Make all of the channel issues compatible
        if channels_in == 6:
            normal_channel = True
            normal_prefix = "normals_"
        elif channels_in == 3:
            normal_channel = False
            normal_prefix = ""
        else:
            normal_channel = None
            normal_prefix = None
        # Initialize the right feature extractor
        if extractor_backbone == "PointNet":
            weights_file_path = f"./drl_grasping/drl_octree/features_extractor/pointnet_{normal_prefix}pretrained.pth"
            self._extractor_backbone = PointNetFeatureExtractor(normal_channel=normal_channel, features_dim=features_dim, file_path=weights_file_path)
        elif extractor_backbone == "PointNet2":
            weights_file_path = f"./drl_grasping/drl_octree/features_extractor/pointnet2_{normal_prefix}pretrained.pth"
            self._extractor_backbone = PointNet2FeatureExtractor(normal_channel=normal_channel, features_dim=features_dim, file_path=weights_file_path)

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
    # Input Parameters
    num_frames = 2
    num_channels = 7
    num_points = 1024
    proprioceptive_observations = True
    # Model Input
    pointcloud_input = torch.randn((num_frames, num_points, num_channels), dtype=torch.float32)
    if proprioceptive_observations:
        aux_dim = 10
        aux_input = torch.randn((1, num_frames, aux_dim), dtype=torch.float32)
    else:
        aux_dim = 0
    # Model Observation Space
    mySpace = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2, num_points+aux_dim, num_channels),dtype=np.float32)
    # Defined Feature Extractor
    MyExtractor = PointCloudCnnFeaturesExtractor(channels_in=num_channels,
                                                 observation_space=mySpace, 
                                                 features_dim=248,
                                                 aux_obs_dim=10,
                                                 aux_obs_features_dim=8,
                                                 extractor_backbone="PointNet",
                                                 )
    # Data output
    data = MyExtractor((pointcloud_input, aux_input))
    print("Final Output: ", data.shape)
