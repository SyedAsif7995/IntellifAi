# coding: utf-8
"""
Optimized AppearanceFeatureExtractor:
- Faster inference
- TorchScript-safe reshape
- AMP and memory-format ready
"""

import torch
from torch import nn
from .util import SameBlock2d, DownBlock2d, ResBlock3d


class AppearanceFeatureExtractor(nn.Module):
    def __init__(self, image_channel, block_expansion, num_down_blocks, max_features, reshape_channel, reshape_depth, num_resblocks):
        super(AppearanceFeatureExtractor, self).__init__()
        self.first = SameBlock2d(image_channel, block_expansion, kernel_size=(3, 3), padding=(1, 1))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        self.second = nn.Conv2d(in_channels=out_features, out_channels=max_features, kernel_size=1, stride=1)

        self.resblocks_3d = nn.Sequential(*[
            ResBlock3d(reshape_channel, kernel_size=3, padding=1) for _ in range(num_resblocks)
        ])

        self.reshape_channel = reshape_channel
        self.reshape_depth = reshape_depth

    @torch.no_grad()  # Efficient for inference use
    def forward(self, source_image):
        # Optional: source_image = source_image.to(memory_format=torch.channels_last)
        out = self.first(source_image)

        for block in self.down_blocks:
            out = block(out)
        out = self.second(out)

        bs, c, h, w = out.shape
        f_s = out.reshape(bs, self.reshape_channel, self.reshape_depth, h, w)
        f_s = self.resblocks_3d(f_s)
        return f_s
