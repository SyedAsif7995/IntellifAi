# coding: utf-8

"""
Optimized DenseMotionNetwork with streamlined 3D processing and modularity
"""

import torch
from torch import nn
import torch.nn.functional as F
from .util import Hourglass, make_coordinate_grid, kp2gaussian

class DenseMotionNetwork(nn.Module):
    def __init__(self, block_expansion, num_blocks, max_features, num_kp, feature_channel, reshape_depth, compress, estimate_occlusion_map=True):
        super(DenseMotionNetwork, self).__init__()

        # 3D Hourglass backbone
        self.hourglass = Hourglass(
            block_expansion=block_expansion,
            in_features=(num_kp + 1) * (compress + 1),
            max_features=max_features,
            num_blocks=num_blocks
        )

        self.mask = nn.Conv3d(self.hourglass.out_filters, num_kp + 1, kernel_size=1)
        self.compress = nn.Sequential(
            nn.Conv3d(feature_channel, compress, kernel_size=1),
            nn.BatchNorm3d(compress),
            nn.ReLU(inplace=True)
        )

        self.num_kp = num_kp
        self.flag_estimate_occlusion_map = estimate_occlusion_map
        self.occlusion = nn.Conv2d(self.hourglass.out_filters * reshape_depth, 1, kernel_size=3, padding=1) if estimate_occlusion_map else None

    def create_sparse_motions(self, feature, kp_driving, kp_source):
        bs, _, d, h, w = feature.shape
        identity_grid = make_coordinate_grid((d, h, w), ref=kp_source).view(1, 1, d, h, w, 3).to(feature.device)
        coordinate_grid = identity_grid - kp_driving.view(bs, self.num_kp, 1, 1, 1, 3)
        driving_to_source = coordinate_grid + kp_source.view(bs, self.num_kp, 1, 1, 1, 3)
        sparse_motions = torch.cat([identity_grid.repeat(bs, 1, 1, 1, 1, 1), driving_to_source], dim=1)
        return sparse_motions

    def create_deformed_feature(self, feature, sparse_motions):
        bs, _, d, h, w = feature.shape
        f = feature.unsqueeze(1).repeat(1, self.num_kp + 1, 1, 1, 1, 1)
        f = f.view(bs * (self.num_kp + 1), -1, d, h, w)
        grid = sparse_motions.view(bs * (self.num_kp + 1), d, h, w, 3)
        deformed = F.grid_sample(f, grid, align_corners=False)
        return deformed.view(bs, self.num_kp + 1, -1, d, h, w)

    def create_heatmap_representations(self, feature, kp_driving, kp_source):
        spatial_size = feature.shape[2:]
        heatmap = kp2gaussian(kp_driving, spatial_size, 0.01) - kp2gaussian(kp_source, spatial_size, 0.01)
        zeros = torch.zeros(heatmap.size(0), 1, *spatial_size, device=feature.device, dtype=feature.dtype)
        return torch.cat([zeros, heatmap], dim=1).unsqueeze(2)

    def forward(self, feature, kp_driving, kp_source):
        bs, _, d, h, w = feature.shape

        feature = self.compress(feature)
        sparse_motion = self.create_sparse_motions(feature, kp_driving, kp_source)
        deformed_feature = self.create_deformed_feature(feature, sparse_motion)
        heatmap = self.create_heatmap_representations(deformed_feature, kp_driving, kp_source)

        input = torch.cat([heatmap, deformed_feature], dim=2).view(bs, -1, d, h, w)
        prediction = self.hourglass(input)

        mask = F.softmax(self.mask(prediction), dim=1)
        sparse_motion = sparse_motion.permute(0, 1, 5, 2, 3, 4)
        deformation = (sparse_motion * mask.unsqueeze(2)).sum(dim=1).permute(0, 2, 3, 4, 1)

        out_dict = {'mask': mask, 'deformation': deformation}

        if self.flag_estimate_occlusion_map:
            reshaped = prediction.view(bs, -1, h, w)
            out_dict['occlusion_map'] = torch.sigmoid(self.occlusion(reshaped))

        return out_dict
