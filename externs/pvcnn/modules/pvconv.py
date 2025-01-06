import torch.nn as nn

import externs.pvcnn.modules.functional as F
from externs.pvcnn.modules.voxelization import Voxelization
from externs.pvcnn.modules.shared_mlp import SharedMLP
from externs.pvcnn.modules.se import SE3d

__all__ = ['PVConv']


class PVConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, resolution, with_se=False, normalize=True, eps=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = resolution

        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        voxel_layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
         ]
        if with_se:
            voxel_layers.append(SE3d(out_channels))
        self.voxel_layers = nn.Sequential(*voxel_layers)
        self.point_features = SharedMLP(in_channels, out_channels)

    def forward(self, inputs):
        features, coords = inputs
        voxel_features, voxel_coords = self.voxelization(features, coords)
        voxel_features = self.voxel_layers(voxel_features)
        voxel_features = F.trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)
        fused_features = voxel_features + self.point_features(features)
        return fused_features, coords

class ProxyVoxelConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, resolution, with_se=False, normalize=True, eps=0):
        super().__init__()
        self.in_channels = in_channels
        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        # self.expansion_layer = nn.Conv3d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)
        # self.expansion_layer.weight.requires_grad=False
        # self.expansion_layer.weight[...] = 1
        # self.expansion_layer = nn.MaxPool3d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)

    def forward(self, inputs):
        features, coords = inputs
        voxel_features, voxel_coords = self.voxelization(features, coords)
        # voxel_features = self.expansion_layer(voxel_features)
        return voxel_features, voxel_coords
    

