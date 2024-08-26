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


class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True, feature_transform=True, k=7):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(k=k)
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
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
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetCls(nn.Module):
    def __init__(self, num_classes=40, normal_channel=True, feature_transform=True):
        super(PointNetCls, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform, k=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


class PointNetFeatureExtractor(nn.Module):
    def __init__(self, normal_channel=True, feature_transform=True, features_dim=248, file_path="./drl_grasping/drl_octree/features_extractor/pointnet_pretrained.pth"):
        super(PointNetFeatureExtractor, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform, k=channel)    
        # load weight dictionary remove unexpected / unused prefixes & items
        state_dict = torch.load(file_path)['model_state_dict']
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
        x = x.permute(0, 2, 1)
        x, _, _ = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    # Set the device for the models
    DEVICE = 'cuda'
    # decide if normals get used
    USE_NORMALS = False
    if USE_NORMALS:
        num_channels = 6
        file_name = "pointnet_normals_pretrained"
    else:
        num_channels = 3
        file_name = "pointnet_pretrained"
    # Get the weights from the state dictionary
    base_init_path = os.path.abspath("../../../../drl_grasping")
    file_path = f"./drl_grasping/drl_octree/features_extractor/{file_name}.pth"
    file_path = os.path.join(base_init_path, file_path)
    print(file_path)
    state_dict = torch.load(file_path)['model_state_dict']
    # Input from observation space
    pointcloud = torch.rand(32, 1024, num_channels)
    # Transposed input for base networks
    sim_data_kd = Variable(pointcloud.permute(0, 2, 1)).to(DEVICE)
    print('Input Data: ', sim_data_kd.size(), "   CUDA: ", sim_data_kd.is_cuda)

    # Getting classes for each point cloud
    classifier = PointNetCls(num_classes=40, normal_channel=USE_NORMALS).to(DEVICE)
    classifier.load_state_dict(state_dict)
    cls, _, _ = classifier(sim_data_kd)
    print('Classes: ', cls.size())
    
    # Getting global features from pretrained base model
    feat_extractor = PointNetfeat(k=num_channels).to(DEVICE)
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
    pointfeat = PointNetfeat(global_feat=False, k=num_channels).to(DEVICE)
    points, _, _ = pointfeat(sim_data_kd)
    print('Point Features:', points.size())
    
    # Getting drl-features from observation space input
    feat_drl = PointNetFeatureExtractor(normal_channel=USE_NORMALS, features_dim=256, file_path=file_path).to(DEVICE)
    out = feat_drl(pointcloud.to(DEVICE))
    print('Feature Extractor: ', out.size())
