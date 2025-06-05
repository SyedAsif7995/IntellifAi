# coding: utf-8
"""
Optimized ConvNeXtV2 for fast inference of implicit keypoints, pose, and expression deformation
"""

import torch
import torch.nn as nn
from .util import LayerNorm, DropPath, trunc_normal_, GRN


class Block(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return residual + self.drop_path(x)


class ConvNeXtV2(nn.Module):
    def __init__(self, in_chans=3, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., **kwargs):
        super().__init__()
        self.downsample_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
            )
        ])
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            ))

        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(4):
            self.stages.append(nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            ))
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        num_bins = kwargs.get('num_bins', 66)
        num_kp = kwargs.get('num_kp', 24)

        self.fc_kp = nn.Linear(dims[-1], 3 * num_kp)
        self.fc_scale = nn.Linear(dims[-1], 1)
        self.fc_pitch = nn.Linear(dims[-1], num_bins)
        self.fc_yaw = nn.Linear(dims[-1], num_bins)
        self.fc_roll = nn.Linear(dims[-1], num_bins)
        self.fc_t = nn.Linear(dims[-1], 3)
        self.fc_exp = nn.Linear(dims[-1], 3 * num_kp)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        x = x.flatten(2).mean(-1)  # global average pooling
        x = self.norm(x)

        return {
            'kp': self.fc_kp(x),
            'pitch': self.fc_pitch(x),
            'yaw': self.fc_yaw(x),
            'roll': self.fc_roll(x),
            't': self.fc_t(x),
            'exp': self.fc_exp(x),
            'scale': self.fc_scale(x)
        }


def convnextv2_tiny(**kwargs):
    return ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
