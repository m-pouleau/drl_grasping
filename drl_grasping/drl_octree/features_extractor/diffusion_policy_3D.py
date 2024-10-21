import sys
sys.path.append("../../../../drl_grasping")
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable

class DP3Extractor(nn.Module):
    def __init__(self, color_channels=3, num_channels=9, features_dim=248, device='cpu'):
        super(DP3Extractor, self).__init__()
        self.used_channels = 6 if color_channels == 3 else 3
        self.num_channels = num_channels
        print(self.block_channels)
        self.block_channels = [64, 128, 256, 512]
        self.use_layernorm = True

        self.mlp = nn.Sequential(
            nn.Linear(self.used_channels, self.block_channels[0]),
            nn.LayerNorm(self.block_channels[0]) if self.use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(self.block_channels[0], self.block_channels[1]),
            nn.LayerNorm(self.block_channels[1]) if self.use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(self.block_channels[1], self.block_channels[2]),
            nn.LayerNorm(self.block_channels[2]) if self.use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(self.block_channels[2], self.block_channels[3]),
        )

        if self.use_layernorm:
            self.final_projection = nn.Sequential(
                nn.Linear(self.block_channels[-1], features_dim),
                nn.LayerNorm(features_dim)
            )
        else:
            self.final_projection = nn.Linear(self.block_channels[-1], features_dim)

    def forward(self, x_obs):
    	# Extracting point-wise feature like in segmentation network
        xyz_coords = x_obs[:, self.num_channels:, :] # xyz_coords is of shape (k, n, 3) - XYZ coordinates of points
        if self.used_channels == 6:
            rgb_colors = x_obs[:, 3:6, :] # rgb_colors is of shape (k, n, 3) - RGB color values of points
            # Concatenate the XYZ coordinates with RGB channels from observation
            x = torch.cat([xyz_coords, rgb_colors], dim=1)  # Shape: (k, n, 6)
        else:
            x = xyz_coords  # Shape: (k, n, 3)

        # Getting point-wise features with mlp extractor
        x = self.mlp(x) # Shape: (k, n, 512)

        # Max Pooling over points -> get strongest local features (good for edge detection)
        x = torch.max(x, dim=1)[0]  # Shape: (k, 512)

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
    feat_drl = DP3Extractor(color_channels=COLOR_CHANNELS, features_dim=256, device=DEVICE).to(DEVICE)
    out = feat_drl(pointcloud)
    print('Feature Extractor: ', out.size())
