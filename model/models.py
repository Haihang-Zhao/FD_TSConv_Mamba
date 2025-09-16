# models.py
import torch
import torch.nn as nn
from mamba_ssm import Mamba

class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int, out_dim: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2))
        )
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):  # x: (B, T, C, H, W)
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        x = self.cnn(x).view(B * T, -1)
        x = self.fc(x)
        return x.view(B, T, -1)  # (B, T, out_dim)


class BiMambaModel(nn.Module):
    def __init__(self, in_channels: int, cnn_out_dim: int, mamba_dim: int, num_classes: int):
        super().__init__()
        self.encoder = CNNEncoder(in_channels, cnn_out_dim)
        self.project = nn.Linear(cnn_out_dim, mamba_dim) if cnn_out_dim != mamba_dim else nn.Identity()
        self.mamba_forward = Mamba(d_model=mamba_dim)
        self.mamba_backward = Mamba(d_model=mamba_dim)
        self.classifier = nn.Sequential(
            nn.Linear(2 * mamba_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):  # x: (B, T, C, H, W)
        x = self.encoder(x)              # (B, T, cnn_out_dim)
        x = self.project(x)              # (B, T, mamba_dim)
        fwd = self.mamba_forward(x)      # (B, T, mamba_dim)
        bwd = torch.flip(self.mamba_backward(torch.flip(x, [1])), [1])
        x = torch.cat([fwd, bwd], dim=-1)  # (B, T, 2*mamba_dim)
        pooled = x.mean(dim=1)           # (B, 2*mamba_dim)
        return self.classifier(pooled)   # (B, num_classes)
