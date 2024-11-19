import sys
sys.path.append("../../../../drl_grasping")
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable

class DP3Extractor(nn.Module):
    def __init__(self, color_channels=3, num_channels=9, features_dim=248):
        super(DP3Extractor, self).__init__()
        self.used_channels = 6 if color_channels == 3 else 3
        self.num_channels = num_channels
        self.block_channels = [64, 128, 256]
        self.use_layernorm = True

        layers = []
        input_dim = self.used_channels

        for output_dim in self.block_channels:
            layers.append(nn.Linear(input_dim, output_dim))
            if self.use_layernorm:
                layers.append(nn.LayerNorm(output_dim))
            layers.append(nn.ReLU())
            input_dim = output_dim  # Update input_dim for the next layer

        self.mlp = nn.Sequential(*layers)

        if self.use_layernorm:
            self.final_projection = nn.Sequential(
                nn.Linear(self.block_channels[-1]*2, features_dim),
                nn.LayerNorm(features_dim)
            )
        else:
            self.final_projection = nn.Linear(self.block_channels[-1]*2, features_dim)

    def forward(self, x_obs):
    	# Extracting point-wise feature like in segmentation network
        xyz_coords = x_obs[:, :, self.num_channels:] # xyz_coords is of shape (k, n, 3) - XYZ coordinates of points
        if self.used_channels == 6:
            rgb_colors = x_obs[:, :, 3:6] # rgb_colors is of shape (k, n, 3) - RGB color values of points
            # Concatenate the XYZ coordinates with RGB channels from observation
            x = torch.cat([xyz_coords, rgb_colors], dim=2)  # Shape: (k, n, 6)
        else:
            x = xyz_coords  # Shape: (k, n, 3)

        # Getting point-wise features with mlp extractor
        x = self.mlp(x) # Shape: (k, n, 512)

        # Max Pooling over points -> get strongest local features (good for edge detection)
        max_pool = torch.max(x, dim=1)[0]  # Shape: (k, 256)
        # Average Pooling over points -> get understanding of global features (overall understanding of scene)
        avg_pool = torch.mean(x, dim=1)  # Shape: (k, 256)
        # Concatenate max pooled and average pooled features along the last dimension
        x = torch.cat([max_pool, avg_pool], dim=1)  # Shape: (k, 512)

        # Apply final layer after pooling                
        x = self.final_projection(x)  # Shape: (k, 248)

        return x


if __name__ == '__main__':
    # Set the device for the models
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'
    print("Device: ", DEVICE)
    # decide if normals and colors get used
    USE_COLORS = True
    if USE_COLORS:
        COLOR_CHANNELS = 3
    else:
        COLOR_CHANNELS = 0
    # Input from observation space
    NUM_CHANNELS = 12
    BATCH_SIZE = 32
    NUM_POINTS = 1024
    # Input from observation space
    pointcloud = torch.rand(BATCH_SIZE, NUM_POINTS, NUM_CHANNELS).to(DEVICE)
    # Transposed input for base networks
    print('Input Data: ', pointcloud.size(), "   CUDA: ", pointcloud.is_cuda)

    # Getting drl-features from observation space input
    feat_drl = DP3Extractor(color_channels=COLOR_CHANNELS, features_dim=256).to(DEVICE)
    out = feat_drl(pointcloud)
    print('Feature Extractor: ', out.size())
