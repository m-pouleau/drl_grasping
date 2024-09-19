import sys
sys.path.append("../../../../drl_grasping")
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from drl_grasping.drl_octree.features_extractor.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation
from drl_grasping.drl_octree.features_extractor.modules import remove_prefix, delete_items_without_prefix


class PointNet2Classification(nn.Module):
    def __init__(self, num_classes=40, normal_channel=True):
        super(PointNet2Classification, self).__init__()
        self.in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], self.in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:6, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
            xyz = xyz[:, :3, :]
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)
        return x, l3_points


class PointNet2Segmentation(nn.Module):
    def __init__(self, num_classes=13):
        super(PointNet2Segmentation, self).__init__()

        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], 9, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]])
        self.fp4 = PointNetFeaturePropagation(512+512+256+256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l4_points


class PointNet2FeatureExtractor(nn.Module):
    def __init__(self, num_channels=9, features_dim=248, device='cpu', file_path=".."):
        super(PointNet2FeatureExtractor, self).__init__()
        self.in_channel = num_channels
        if self.in_channel != 9:
            self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], self.in_channel-3,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
            self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
            self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        else:
            self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], self.in_channel, [[16, 16, 32], [32, 32, 64]])
            self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]])
            self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]])

        # load weight dictionary remove unexpected / unused prefixes & items
        state_dict = torch.load(file_path, map_location=torch.device(device))['model_state_dict']
        state_dict_1 = delete_items_without_prefix(state_dict.copy(), 'sa1.')
        state_dict_2 = delete_items_without_prefix(state_dict.copy(), 'sa2.')
        state_dict_3 = delete_items_without_prefix(state_dict.copy(), 'sa3.')
        state_dict_1 = remove_prefix(state_dict_1, 'sa1.')
        state_dict_2 = remove_prefix(state_dict_2, 'sa2.')
        state_dict_3 = remove_prefix(state_dict_3, 'sa3.')
        self.sa1.load_state_dict(state_dict_1)
        self.sa2.load_state_dict(state_dict_2)
        self.sa3.load_state_dict(state_dict_3)
        # freeze weights of pretrained model
        for module in [self.sa1, self.sa2, self.sa3]:
            for param in module.parameters():
                param.requires_grad = False        
        # Additional unfrozen linear layers
        if self.in_channel != 9:
            self.fc1 = nn.Linear(1024, 512)
        else:
            self.fc1 = nn.Linear(512, 512)

        self.fc2 = nn.Linear(512, features_dim)
        self.bn1 = nn.BatchNorm1d(512)

    def forward(self, pointcloud):
        xyz = pointcloud.permute(0, 2, 1)
        B, _, _ = xyz.shape

        if self.in_channel == 9:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        elif self.in_channel == 6:
            l0_points = xyz[:, 3:6, :]
            l0_xyz = xyz[:, :3, :]
        else:
            l0_points = None
            l0_xyz = xyz[:, :3, :]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        if self.in_channel != 9:
            x = l3_points.view(B, 1024)
        else:
            x, _ = torch.max(l3_points, 2)

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    # Set the device for the models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'
    print("Device: ", DEVICE)
    # decide if normals and colors get used
    USE_COLORS = True
    USE_NORMALS = False
    if USE_COLORS:
        NUM_CHANNELS = 9
        file_name = "pointnet2_msg_seg_pretrained"
    elif USE_NORMALS:
        NUM_CHANNELS = 6
        file_name = "pointnet2_msg_normals_pretrained"
    else:
        NUM_CHANNELS = 3
        file_name = "pointnet2_msg_pretrained"
    # Input from observation space
    BATCH_SIZE = 8
    NUM_POINTS = 1024
    # Get the weights from the state dictionary
    script_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = f"{script_directory}/weights/{file_name}.pth"
    state_dict = torch.load(file_path, map_location=torch.device(DEVICE))['model_state_dict']
    # Input from observation space
    pointcloud = torch.rand(BATCH_SIZE, NUM_POINTS, NUM_CHANNELS)
    # Transposed input for base networks
    sim_data_kd = Variable(pointcloud.permute(0, 2, 1)).to(DEVICE)
    print('Input Data: ', sim_data_kd.size(), "   CUDA: ", sim_data_kd.is_cuda)

    # Getting classes for each point cloud / for each point
    if not USE_COLORS:
        classifier = PointNet2Classification(num_classes=40, normal_channel=USE_NORMALS).to(DEVICE)
        classifier.load_state_dict(state_dict)
        cls, _ = classifier(sim_data_kd)
        print('Object Classes: ', cls.size())
    else:
        segmenter = PointNet2Segmentation(num_classes=13).to(DEVICE)
        #segmenter.load_state_dict(state_dict)
        out, _ = segmenter(sim_data_kd)
        print('Point Classes: ', out.size())

    # Getting drl-features from observation space input
    feat_drl = PointNet2FeatureExtractor(num_channels=NUM_CHANNELS, features_dim=256, file_path=file_path, device=DEVICE).to(DEVICE)
    out = feat_drl(pointcloud.to(DEVICE))
    print('Feature Extractor: ', out.size())
