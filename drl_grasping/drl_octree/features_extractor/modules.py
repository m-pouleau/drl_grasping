import ocnn
import torch


class OctreeConvRelu(torch.nn.Module):
    def __init__(self, depth, channel_in, channel_out, kernel_size=[3], stride=1):
        super(OctreeConvRelu, self).__init__()
        self.conv = ocnn.OctreeConv(depth, channel_in, channel_out, kernel_size, stride)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data_in, octree):
        out = self.conv(data_in, octree)
        out = self.relu(out)
        return out


class OctreeConvBnRelu(torch.nn.Module):
    def __init__(
        self,
        depth,
        channel_in,
        channel_out,
        kernel_size=[3],
        stride=1,
        bn_eps=0.00001,
        bn_momentum=0.01,
    ):
        super(OctreeConvBnRelu, self).__init__()
        self.conv = ocnn.OctreeConv(depth, channel_in, channel_out, kernel_size, stride)
        self.bn = torch.nn.BatchNorm2d(channel_out, bn_eps, bn_momentum)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data_in, octree):
        out = self.conv(data_in, octree)
        out = self.bn(out)
        out = self.relu(out)
        return out


class OctreeConvFastRelu(torch.nn.Module):
    def __init__(self, depth, channel_in, channel_out, kernel_size=[3], stride=1):
        super(OctreeConvFastRelu, self).__init__()
        self.conv = ocnn.OctreeConvFast(
            depth, channel_in, channel_out, kernel_size, stride
        )
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data_in, octree):
        out = self.conv(data_in, octree)
        out = self.relu(out)
        return out


class OctreeConvFastBnRelu(torch.nn.Module):
    def __init__(
        self,
        depth,
        channel_in,
        channel_out,
        kernel_size=[3],
        stride=1,
        bn_eps=0.00001,
        bn_momentum=0.01,
    ):
        super(OctreeConvFastBnRelu, self).__init__()
        self.conv = ocnn.OctreeConvFast(
            depth, channel_in, channel_out, kernel_size, stride
        )
        self.bn = torch.nn.BatchNorm2d(channel_out, bn_eps, bn_momentum)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data_in, octree):
        out = self.conv(data_in, octree)
        out = self.bn(out)
        out = self.relu(out)
        return out


class OctreeConv1x1Relu(torch.nn.Module):
    def __init__(self, channel_in, channel_out, use_bias=True):
        super(OctreeConv1x1Relu, self).__init__()
        self.conv1x1 = ocnn.OctreeConv1x1(channel_in, channel_out, use_bias)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data_in):
        out = self.conv1x1(data_in)
        out = self.relu(out)
        return out


class OctreeConv1x1BnRelu(torch.nn.Module):
    def __init__(
        self, channel_in, channel_out, use_bias=True, bn_eps=0.00001, bn_momentum=0.01
    ):
        super(OctreeConv1x1BnRelu, self).__init__()
        self.conv1x1 = ocnn.OctreeConv1x1(channel_in, channel_out, use_bias)
        self.bn = torch.nn.BatchNorm2d(channel_out, bn_eps, bn_momentum)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data_in):
        out = self.conv1x1(data_in)
        out = self.bn(out)
        out = self.relu(out)
        return out


class LinearRelu(torch.nn.Module):
    def __init__(self, channel_in, channel_out, use_bias=True):
        super(LinearRelu, self).__init__()
        self.fc = torch.nn.Linear(channel_in, channel_out, use_bias)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data_in):
        out = self.fc(data_in)
        out = self.relu(out)
        return out


class LinearBnRelu(torch.nn.Module):
    def __init__(
        self, channel_in, channel_out, use_bias=True, bn_eps=0.00001, bn_momentum=0.01
    ):
        super(LinearBnRelu, self).__init__()
        self.fc = torch.nn.Linear(channel_in, channel_out, use_bias)
        self.bn = torch.nn.BatchNorm1d(channel_out, bn_eps, bn_momentum)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data_in):
        out = self.fc(data_in)
        out = self.bn(out)
        out = self.relu(out)
        return out


class ImageConvRelu(torch.nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, stride=1, padding=1):
        super(ImageConvRelu, self).__init__()
        self.conv = torch.nn.Conv2d(
            channel_in, channel_out, kernel_size, stride, padding
        )
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data_in):
        out = self.conv(data_in)
        out = self.relu(out)
        return out


def remove_prefix(dictionary, prefix):
    """
    Given a dictionary, a given prefix is removed from all keys and a new, modified dictionary is returned.
    Can be used to remove unwanted prefixes before loading a state_dict (pretrained model weights)
    Author: Lukas Seitz

    :param dictionary: input dictionary containing undesired prefixes in keys
    :param prefix: string in dictionary keys to be removed
    :returns: dictionary without given prefixes in keys
    """
    new_dictionary = {}
    for key, value in dictionary.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]  # Remove the prefix
        else:
            new_key = key  # Keep the original key
        new_dictionary[new_key] = value
    return new_dictionary


def delete_items_with_prefix(dictionary, prefix):
    """
    Given a dictionary, delete all items starting with a certain prefix.
    Can be used to delete e.g. all entries related to a classifier, which is not needed for feature extraction.
    Author: Lukas Seitz
    """
    keys_to_delete = [key for key in dictionary.keys() if key.startswith(prefix)]
    for key in keys_to_delete:
        del dictionary[key]
    return dictionary


def delete_items_without_prefix(dictionary, prefix):
    """
    Given a dictionary, delete all items that do not start with a certain prefix.
    This can be useful to keep only specific entries in a dictionary.
    """
    keys_to_delete = [key for key in dictionary.keys() if not key.startswith(prefix)]
    for key in keys_to_delete:
        del dictionary[key]
    return dictionary
