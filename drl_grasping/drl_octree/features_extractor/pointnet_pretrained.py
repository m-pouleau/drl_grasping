import sys
sys.path.append("../../../../drl_grasping")
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as grad
from drl_grasping.utils.utils import remove_prefix, delete_items_with_prefix


# Class for Transformer
class Transformer(nn.Module):
    """
    Computes a KxK affine transform from the input data to transform inputs to a "canonical view"

    """
    def __init__(self, K=3):
        super(Transformer, self).__init__()
		# Number of dimensions of the data
        self.K = K
		# Initialize identity matrix on the GPU (do this here so it only happens once)
        self.identity = grad.Variable(torch.eye(self.K).double().view(-1))
		# First embedding block
        self.block1 =nn.Sequential(
			nn.Conv1d(K, 64, 1),
			nn.BatchNorm1d(64),
			nn.ReLU())
		# Second embedding block
        self.block2 =nn.Sequential(
			nn.Conv1d(64, 128, 1),
			nn.BatchNorm1d(128),
			nn.ReLU())
		# Third embedding block
        self.block3 =nn.Sequential(
			nn.Conv1d(128, 1024, 1),
			nn.BatchNorm1d(1024),
			nn.ReLU())
		# Multilayer perceptron
        self.mlp = nn.Sequential(
			nn.Linear(1024, 512),
			nn.BatchNorm1d(512),
			nn.ReLU(),
			nn.Linear(512, 256),
			nn.BatchNorm1d(256),
			nn.ReLU(),
			nn.Linear(256, K * K))

	# Take as input a B x K x N matrix of B batches of N points with K dimensions
    def forward(self, x):
		# Compute the feature extractions: Output should ultimately be B x 1024 x N
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
		# Pool over the number of points: Output should be B x 1024 x 1 --> B x 1024 (after squeeze)
        x = torch.max(x, 2, keepdim=True)[0].view(-1, 1024)
		# Run the pooled features through the multi-layer perceptron: Output should be B x K^2
        x = self.mlp(x)
		# Add identity matrix to transform: Output is still B x K^2 (broadcasting takes care of batch dimension)
        if x.is_cuda:
            self.identity = self.identity.cuda()
        x += self.identity
		# Reshape the output into B x K x K affine transformation matrices
        x = x.view(-1, self.K, self.K)
        return x


# Class for PointNetBase acting as a Feature Extractor
class PointNetBase(nn.Module):
    """
    Computes the local embeddings and global features for an input set of points
    """
    def __init__(self, K=3):
        super(PointNetBase, self).__init__()
		# Input transformer for K-dimensional input, K should be 3 for XYZ coordinates, but can be larger if normals, colors, etc are included
        self.input_transformer = Transformer(K)
		# Embedding transformer is always going to be 64 dimensional
        self.embedding_transformer = Transformer(64)
		# First multilayer perceptron with shared weights implemented as convolutions
        self.mlp1 = nn.Sequential(
			nn.Conv1d(K, 64, 1),
			nn.BatchNorm1d(64),
			nn.ReLU(),
			nn.Conv1d(64, 64, 1),
			nn.BatchNorm1d(64),
			nn.ReLU())
        # Second multilayer perceptron with shared weights implemented as convolutions
        self.mlp2 = nn.Sequential(
			nn.Conv1d(64, 64, 1),
			nn.BatchNorm1d(64),
			nn.ReLU(),
			nn.Conv1d(64, 128, 1),
			nn.BatchNorm1d(128),
			nn.ReLU(),
			nn.Conv1d(128, 1024, 1),
			nn.BatchNorm1d(1024),
			nn.ReLU())

	# Take as input a B x K x N matrix of B batches of N points with K dimensions
    def forward(self, x):
		# Number of points put into the network
        N = x.shape[2]
		# First compute the input data transform: T1 is B x K x K and x output is B x K x N
        T1 = self.input_transformer(x)
        x = torch.bmm(T1, x)
		# Run the transformed inputs through the first embedding MLP: Output is B x 64 x N
        x = self.mlp1(x)
		# Transform the embeddings: T2 is B x 64 x 64 and output is B x 64 x N
        T2 = self.embedding_transformer(x)
        local_embedding = torch.bmm(T2, x)
		# Further embed the "local embeddings": Output is B x 1024 x N
        global_feature = self.mlp2(local_embedding)
		# Pool over the number of points: Output should be B x 1024 (after squeeze) == "global features"
        global_feature = F.max_pool1d(global_feature, N).squeeze(2)
        return global_feature, local_embedding, T2


# Class for PointNet Classification
class PointNetClassifier(nn.Module):
    '''
    Embedding to enable loading of weights
    '''

    def __init__(self, K=3):
        super(PointNetClassifier, self).__init__()
		# Local and global feature extractor for PointNet
        self.base = PointNetBase(K)

        # Classifier for ShapeNet
        self.classifier = nn.Sequential(
			nn.Linear(1024, 512),
			nn.BatchNorm1d(512),
			nn.ReLU(),
			nn.Dropout(0.7),
			nn.Linear(512, 256),
			nn.BatchNorm1d(256),
			nn.ReLU(),
			nn.Dropout(0.7),
			nn.Linear(256, 40))


	# Take as input a B x K x N matrix of B batches of N points with K dimensions
    def forward(self, x):
		# Only need to keep the global feature descriptors for classification: Output should be B x 1024
        x, _, T2 = self.base(x)
		# Returns a B x 40
        return self.classifier(x)


if __name__ == '__main__':
    # Set the device for the models
    device = 'cuda'
    # Get the weights from the state dictionary
    base_init_path = os.path.abspath("../../../../drl_grasping")
    file_path = "./drl_grasping/drl_octree/features_extractor/classifier_model_state.pth"
    file_path = os.path.join(base_init_path, file_path)
    state_dict = torch.load(file_path)    
    # Input from observation space
    pointcloud = torch.rand(128, 2048, 3)
    # Transposed input for base networks
    sim_data_7d = pointcloud.permute(0, 2, 1).to(device)
    print('Input Data: ', sim_data_7d.size(), "   CUDA: ", sim_data_7d.is_cuda)
    
    # Definition of a pretrained feature extractor
    FeatureExtractor = PointNetBase(K=3).to(device)
    # remove unexpected / unused prefixes & items from the loaded dictionary
    new_dict = remove_prefix(state_dict, 'base.')
    new_dict = delete_items_with_prefix(new_dict, 'classifier')
    FeatureExtractor.load_state_dict(new_dict)
    # freeze weights of pretrained model
    for param in FeatureExtractor.parameters():
        param.requires_grad = False
    # perform forward pass on input
    feats, _, _ = FeatureExtractor(sim_data_7d)
    print(feats.shape)
    
    # Definition of a pretrained classifier and forward pass
    Classifier = PointNetClassifier(K=3).to(device)
    Classifier.load_state_dict(state_dict)
    cls = Classifier(sim_data_7d)
    print(cls.shape)
