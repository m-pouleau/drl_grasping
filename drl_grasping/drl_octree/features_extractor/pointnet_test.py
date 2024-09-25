import sys
sys.path.append("../../../../drl_grasping")
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from drl_grasping.drl_octree.features_extractor.modules import remove_prefix, delete_items_without_prefix
from drl_grasping.drl_octree.features_extractor.pointnet import PointNetEncoder, PointNetFeatureExtractor

import time

class NewFeatureExtractor(nn.Module):
    def __init__(self, num_channels=9, features_dim=248, device='cpu', file_path=".."):
        super(NewFeatureExtractor, self).__init__()
        self.channel = num_channels
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, k=self.channel)    
        # load weight dictionary remove unexpected / unused prefixes & items
        state_dict = torch.load(file_path, map_location=torch.device(device))['model_state_dict']
        state_dict = delete_items_without_prefix(state_dict, "feat.")
        state_dict = remove_prefix(state_dict, 'feat.')
        self.feat.load_state_dict(state_dict)
        # freeze weights of pretrained model
        for param in self.feat.parameters():
            param.requires_grad = False

        ## Additional unfrozen linear layers ##
        # Point-wise fully connected layer before pooling
        self.pointwise_fc = nn.Linear(1091, 512)
        
        # Learnable layers after pooling
        self.fc2 = nn.Linear(1024, 512)  # Max pool + Avg pool -> (512 + 512) = 1024
        self.fc3 = nn.Linear(512, features_dim)   # Final layer to reduce to 248 features
        self.bn = nn.BatchNorm1d(512)

    def forward(self, x_obs):
    	# Extracting point-wise feature like in segmentation network
        x = x_obs.permute(0, 2, 1)[:, :self.channel, :]
        xyz_coords = x_obs.permute(0, 2, 1)[:, self.channel:, :] # xyz_coords is of shape (k, n, 3) - XYZ coordinates of points
        x, _, _ = self.feat(x) # x is of shape (k, n, 1088) - PointNet output

        # Concatenate the XYZ coordinates to the PointNet output
        x = torch.cat([x, xyz_coords], dim=1).permute(0, 2, 1)  # Shape: (k, n, 1091)

        # Point-wise fully connected layer
        x = F.relu(self.pointwise_fc(x))  # Shape: (k, n, 512)

        # Max Pooling over points -> get strongest local features (good for edge detection)
        max_pool, _ = torch.max(x, dim=1)  # Shape: (k, 512)
        # Average Pooling over points -> get understanding of global features (overall understanding of scene)
        avg_pool = torch.mean(x, dim=1)  # Shape: (k, 512)
        # Concatenate max pooled and average pooled features along the last dimension
        x = torch.cat([max_pool, avg_pool], dim=1)  # Shape: (k, 1024)

        # Apply learnable layers after pooling
        x = F.relu(self.bn(self.fc2(x)))  # Shape: (k, 512)
                
        x = self.fc3(x)  # Shape: (k, 248)

        return x


if __name__ == '__main__':
    # Set the device for the models
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'
    print("Device: ", DEVICE)
    # decide if normals and colors get used
    USE_COLORS = False
    USE_NORMALS = False
    if USE_COLORS:
        NUM_CHANNELS = 9
        file_name = "pointnet_seg_pretrained"
    elif USE_NORMALS:
        NUM_CHANNELS = 6
        file_name = "pointnet_normals_pretrained"
    else:
        NUM_CHANNELS = 3
        file_name = "pointnet_pretrained"
    # Input from observation space
    BATCH_SIZE = 128
    NUM_POINTS = 1024
    # Get the weights from the state dictionary
    script_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = f"{script_directory}/weights/{file_name}.pth"
    
    # Create new feature extractors
    feat_pointnet = PointNetFeatureExtractor(num_channels=NUM_CHANNELS, features_dim=248, file_path=file_path, device=DEVICE).to(DEVICE)
    feat_new = NewFeatureExtractor(num_channels=NUM_CHANNELS, features_dim=248, file_path=file_path, device=DEVICE).to(DEVICE)
    
    NUM_FW_PASSES = 100
    total_time = 0.0
    for i in range(NUM_FW_PASSES):
        pointcloud = torch.rand(BATCH_SIZE, NUM_POINTS, NUM_CHANNELS+3)
        start = time.time()
        out = feat_pointnet(pointcloud.to(DEVICE))
        end = time.time()
        total_time += end -start
    
    print('PointNet Feature Extractor avg. time: ', total_time/NUM_FW_PASSES)

    total_time2 = 0.0
    for i in range(NUM_FW_PASSES):
        pointcloud = torch.rand(BATCH_SIZE, NUM_POINTS, NUM_CHANNELS+3)
        start = time.time()
        out = feat_new(pointcloud.to(DEVICE))
        end = time.time()
        total_time2 += end -start
    
    print('New Feature Extractor avg. time: ', total_time2/NUM_FW_PASSES)
