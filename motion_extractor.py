# coding: utf-8

"""
Motion extractor (M): predicts canonical keypoints, head pose, expression deformation from input image
"""

import torch
from torch import nn
from .convnextv2 import convnextv2_tiny
from .util import filter_state_dict

_model_map = {
    'convnextv2_tiny': convnextv2_tiny,
}

class MotionExtractor(nn.Module):
    def __init__(self, backbone='convnextv2_tiny', **kwargs):
        super().__init__()
        self.detector = _model_map.get(backbone, convnextv2_tiny)(**kwargs)

    def load_pretrained(self, path: str):
        if path:
            state = torch.load(path, map_location='cpu')['model']
            filtered = filter_state_dict(state, remove_name='head')
            ret = self.detector.load_state_dict(filtered, strict=False)
            print(f'Loaded pretrained weights from {path} with status: {ret}')

    def forward(self, x):
        return self.detector(x)
