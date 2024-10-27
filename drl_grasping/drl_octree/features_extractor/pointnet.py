import sys
sys.path.append("../../../../drl_grasping")
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from drl_grasping.drl_octree.features_extractor.modules import remove_prefix, delete_items_without_prefix


class STN3d(nn.Module):
    def __init__(self, k=7):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=7):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, point_feat=False, feature_transform=True, k=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(k=k)
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.point_feat = point_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.point_feat:
            x_tile = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x_tile, pointfeat], 1), x, pointfeat
        else:
            return x, trans, trans_feat


class PointNetCls(nn.Module):
    def __init__(self, num_classes=40, normal_channel=False):
        super(PointNetCls, self).__init__()
        if normal_channel:
            self.channel = 6
        else:
            self.channel = 3
        self.feat = PointNetEncoder(point_feat=False, feature_transform=True, k=self.channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x, trans, trans_feat = self.feat(x[:, :self.channel, :])
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


class PointNetSeg(nn.Module):
    def __init__(self, num_classes=13):
        super(PointNetSeg, self).__init__()
        self.k = num_classes
        self.feat = PointNetEncoder(point_feat=True, feature_transform=True, k=9)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, _, _ = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x


class PointNetFeatureExtractorSimple(nn.Module):
    def __init__(self, num_channels=9, features_dim=248, device='cpu', file_path=".."):
        super(PointNetFeatureExtractorSimple, self).__init__()
        self.channel = num_channels
        self.feat = PointNetEncoder(point_feat=False, feature_transform=True, k=self.channel)    
        # load weight dictionary remove unexpected / unused prefixes & items
        state_dict = torch.load(file_path, map_location=torch.device(device))['model_state_dict']
        state_dict = delete_items_without_prefix(state_dict, "feat.")
        state_dict = remove_prefix(state_dict, 'feat.')
        self.feat.load_state_dict(state_dict)
        # freeze weights of pretrained model
        for param in self.feat.parameters():
            param.requires_grad = False
        # Additional unfrozen linear layers
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, features_dim)
        self.bn1 = nn.BatchNorm1d(512)

    def forward(self, x):
        x = x.permute(0, 2, 1)[:, :self.channel, :]
        x, _, _ = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        return x


class PointNetFeatureExtractor(nn.Module):
    def __init__(self, num_channels=9, features_dim=248, device='cpu', file_path=".."):
        super(PointNetFeatureExtractor, self).__init__()
        self.channel = num_channels
        self.feat = PointNetEncoder(point_feat=True, feature_transform=True, k=self.channel)    
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
        if self.channel == 9:
            self.pointwise_fc = nn.Linear(70, 128)
        else:
            self.pointwise_fc = nn.Linear(67, 128)
        self.ln_pw = nn.LayerNorm(128)

        # Fully connected layer for global features
        self.global_fc = nn.Linear(1024, 256)
        self.ln_gl = nn.LayerNorm(256)

        # Learnable layers after pooling
        self.fc3 = nn.Linear(512, features_dim)  # Max pool + Avg pool + Global -> 512 | # Final layer to reduce to 248 features

    def forward(self, x_obs):
    	# Extracting point-wise feature like in segmentation network
        x = x_obs.permute(0, 2, 1)[:, :self.channel, :]
        xyz_coords = x_obs.permute(0, 2, 1)[:, self.channel:, :] # xyz_coords is of shape (k, n, 3) - XYZ coordinates of points
        if self.channel == 9:
            rgb_colors = x_obs.permute(0, 2, 1)[:, 3:6, :] # rgb_colors is of shape (k, n, 3) - RGB color values of points

        _, global_x, pointwise_x = self.feat(x) # x is of shape (k, 1024) & (k, 64, n) - PointNet output

        # Fully connected layer for global features
        global_x = F.relu(self.ln_gl(self.global_fc(global_x)))  # Shape: (k, 320)

        # Concatenate the XYZ coordinates to the PointNet output
        if self.channel == 9:
            pointwise_x = torch.cat([pointwise_x, xyz_coords, rgb_colors], dim=1).permute(0, 2, 1)  # Shape: (k, n, 70)
        else:
            pointwise_x = torch.cat([pointwise_x, xyz_coords], dim=1).permute(0, 2, 1)  # Shape: (k, n, 67)

        # Point-wise fully connected layer
        pointwise_x = F.relu(self.ln_pw(self.pointwise_fc(pointwise_x)))  # Shape: (k, n, 128)

        # Max Pooling over points -> get strongest local features (good for edge detection)
        max_pool, _ = torch.max(pointwise_x, dim=1)  # Shape: (k, 128)
        # Average Pooling over points -> get understanding of global features (overall understanding of scene)
        avg_pool = torch.mean(pointwise_x, dim=1)  # Shape: (k, 128)

        # Concatenate max pooled and average pooled features, as well as global features, along the last dimension
        x = torch.cat([max_pool, avg_pool, global_x], dim=1)  # Shape: (k, 512)

        # Apply learnable layer after pooling                
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
    USE_COLORS = True
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
    BATCH_SIZE = 32
    NUM_POINTS = 1024
    # Get the weights from the state dictionary
    script_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = f"{script_directory}/weights/{file_name}.pth"
    state_dict = torch.load(file_path, map_location=torch.device(DEVICE))['model_state_dict']
    # Input from observation space
    pointcloud = torch.rand(BATCH_SIZE, NUM_POINTS, NUM_CHANNELS+3)
    # Transposed input for base networks
    sim_data_kd = Variable(pointcloud[:, :, :NUM_CHANNELS].permute(0, 2, 1)).to(DEVICE)
    print('Input Data: ', sim_data_kd.size(), "   CUDA: ", sim_data_kd.is_cuda)

    # Getting classes for each point cloud / for each point
    if not USE_COLORS:
        classifier = PointNetCls(num_classes=40, normal_channel=USE_NORMALS).to(DEVICE)
        classifier.load_state_dict(state_dict)
        cls, _, _ = classifier(sim_data_kd)
        print('Object Classes: ', cls.size())
    else:
        segmenter = PointNetSeg(num_classes=13).to(DEVICE)
        segmenter.load_state_dict(state_dict)
        out = segmenter(sim_data_kd)
        print('Point Classes: ', out.size())

    # Getting global features from pretrained base model
    feat_extractor = PointNetEncoder(k=NUM_CHANNELS).to(DEVICE)
    # remove unexpected / unused prefixes & items from the loaded dictionary
    new_dict = delete_items_without_prefix(state_dict, "feat.")
    new_dict = remove_prefix(new_dict, 'feat.')
    feat_extractor.load_state_dict(new_dict)
    # freeze weights of pretrained model
    for param in feat_extractor.parameters():
        param.requires_grad = False
    glob, _, _ = feat_extractor(sim_data_kd)
    print('Global Features:', glob.size())
    
    # Getting point features and local features
    pointfeat = PointNetEncoder(point_feat=True, k=NUM_CHANNELS).to(DEVICE)
    points, _, _ = pointfeat(sim_data_kd)
    print('Point Features:', points.size())

    # Getting drl-features from observation space input
    feat_drl = PointNetFeatureExtractor(num_channels=NUM_CHANNELS, features_dim=256, file_path=file_path, device=DEVICE).to(DEVICE)
    out = feat_drl(pointcloud.to(DEVICE))
    print('Feature Extractor: ', out.size())
