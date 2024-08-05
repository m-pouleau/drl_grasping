import torch
import torch.nn as nn
import torch.nn.functional as F
from drl_grasping.drl_octree.features_extractor.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction


class PointNet2Classification(nn.Module):
    def __init__(self,num_class, in_channel):
        super(PointNet2Classification, self).__init__()
        used_features_01 = in_channel - 3
        used_features_02 = 314 + in_channel
        used_features_03 = 637 + in_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], used_features_01,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], used_features_02,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, used_features_03, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, pointcloud):
        B, C, _ = pointcloud.shape
        # Split up features of pointcloud
        norm = pointcloud[:, 3:6, :]
        if C > 6: 
            xyz = pointcloud[:, :3, :]
            colors = pointcloud[:, 6:, :]
            points = torch.cat((xyz, colors), dim=1)
        else:
            points = pointcloud[:, :3, :]

        l1_points, l1_norm = self.sa1(points, norm)
        l2_points, l2_norm = self.sa2(l1_points, l1_norm)
        l3_points, l3_norm = self.sa3(l2_points, l2_norm)
        x = l3_norm.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)


        return x,l3_norm


class PointNet2FeatureExtractor(nn.Module):
    def __init__(self, features_dim=248, in_channel=7):
        super(PointNet2FeatureExtractor, self).__init__()
        used_features_01 = in_channel - 3
        used_features_02 = 314 + in_channel
        used_features_03 = 637 + in_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], used_features_01,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], used_features_02,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, used_features_03, [256, 512, 1024], True)
        self.fc = nn.Linear(1024, features_dim)

    def forward(self, pointcloud):
        pointcloud = pointcloud.permute(0, 2, 1)
        B, C, _ = pointcloud.shape
        # Split up features of pointcloud
        norm = pointcloud[:, 3:6, :]
        if C > 6: 
            xyz = pointcloud[:, :3, :]
            colors = pointcloud[:, 6:, :]
            points = torch.cat((xyz, colors), dim=1)
        else:
            points = pointcloud[:, :3, :]

        l1_points, l1_norm = self.sa1(points, norm)
        l2_points, l2_norm = self.sa2(l1_points, l1_norm)
        _, l3_norm = self.sa3(l2_points, l2_norm)
        x = l3_norm.view(B, 1024)
        x = F.relu(self.fc(x))

        return x


if __name__ == '__main__':
    # Input from observation space
    batch_size = 2
    num_points = 2048
    num_dimensions = 7
    pointcloud = torch.rand(batch_size, num_points, num_dimensions)
    # Transposed input for base networks
    sim_data_kd = pointcloud.permute(0, 2, 1)
    print('Input Data: ', sim_data_kd.size())
    # Getting classes for each point cloud
    cls = PointNet2Classification(num_class=11, in_channel=num_dimensions)
    out, _ = cls(sim_data_kd)
    print('Classes: ', out.size())
    # Getting drl-features from observation space input
    feat = PointNet2FeatureExtractor(features_dim=256, in_channel=num_dimensions)
    out = feat(pointcloud)
    print('Feature Extractor: ', out.size())
