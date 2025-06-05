# coding: utf-8

"""
Stitching (S) and Retargeting (R) modules to seamlessly blend animated portraits
and address fine facial motion gaps (eyes, lips) in cross-ID reenactment.
"""

from torch import nn

class StitchingRetargetingNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        layers = []
        in_features = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU(inplace=True))
            in_features = h
        layers.append(nn.Linear(in_features, output_size))
        self.mlp = nn.Sequential(*layers)

    def initialize_weights_to_zero(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.mlp(x)
