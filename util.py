import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# --- Core Blocks ---

class Residual3DBlock(nn.Module):
    def __init__(self, channels, kernel=3, padding=1):
        super().__init__()
        self.norm1 = nn.BatchNorm3d(channels)
        self.conv1 = nn.Conv3d(channels, channels, kernel, padding=padding)
        self.norm2 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel, padding=padding)

    def forward(self, x):
        residual = x
        out = F.relu(self.norm1(x))
        out = self.conv1(out)
        out = F.relu(self.norm2(out))
        out = self.conv2(out)
        return out + residual


class Upsample3D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel, padding=padding)
        self.norm = nn.BatchNorm3d(out_ch)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        x = self.conv(x)
        x = self.norm(x)
        return F.relu(x)


class Downsample2D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=padding)
        self.norm = nn.BatchNorm2d(out_ch)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x)
        return self.pool(x)

# --- SPADE normalization for conditional generation ---

class SPADE(nn.Module):
    def __init__(self, norm_channels, label_channels):
        super().__init__()
        self.norm = nn.InstanceNorm2d(norm_channels, affine=False)
        hidden_dim = 128
        self.shared = nn.Sequential(
            nn.Conv2d(label_channels, hidden_dim, 3, padding=1),
            nn.ReLU())
        self.gamma = nn.Conv2d(hidden_dim, norm_channels, 3, padding=1)
        self.beta = nn.Conv2d(hidden_dim, norm_channels, 3, padding=1)

    def forward(self, x, segmap):
        normalized = self.norm(x)
        segmap = F.interpolate(segmap, size=x.shape[2:], mode='nearest')
        actv = self.shared(segmap)
        gamma = self.gamma(actv)
        beta = self.beta(actv)
        return normalized * (1 + gamma) + beta

# --- Utility: truncated normal init ---

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp > a) & (tmp < b)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
    return tensor

# Example usage for innovative custom layers:
if __name__ == "__main__":
    x = torch.randn(2, 64, 16, 32, 32)  # batch, channels, depth, height, width
    block = Residual3DBlock(64)
    y = block(x)
    print("Residual3DBlock output shape:", y.shape)
