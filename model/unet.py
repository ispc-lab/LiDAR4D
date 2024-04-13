# ==============================================================================
# Copyright (c) 2024 Zehan Zheng. All Rights Reserved.
# LiDAR4D: Dynamic Neural Fields for Novel Space-time View LiDAR Synthesis
# CVPR 2024
# https://github.com/ispc-lab/LiDAR4D
# Apache License 2.0
# ==============================================================================

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    """
    (BN => ReLU => Dropout => Conv) * 2
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout=0.1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling with maxpool then double conv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.down(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    """
    Upscaling then double conv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class AttnBlock(nn.Module):
    """
    Multi-head attention with dropout
    """
    def __init__(self, in_ch, num_head=8, dropout=0.1):
        super().__init__()
        self.proj_qkv = nn.Conv2d(in_ch, in_ch * 3, 1, stride=1, padding=0, bias=False)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0, bias=False)
        self.norm = nn.BatchNorm2d(in_ch)
        self.dropout = dropout
        self.num_head = num_head

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.proj_qkv(h)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # multi-head attention
        q = q.view(B, self.num_head, -1, H * W).permute(0, 1, 3, 2)  # [B, num_head, H * W, C]
        k = k.view(B, self.num_head, -1, H * W)  # [B, num_head, C, H * W]
        v = v.view(B, self.num_head, -1, H * W).permute(0, 1, 3, 2)  # [B, num_head, H * W, C]

        w = torch.matmul(q, k) * (int(C//self.num_head) ** (-0.5))

        # dropout
        if self.training:
            m_r = torch.ones_like(w) * self.dropout
            w = w + torch.bernoulli(m_r) * -1e12

        w = F.softmax(w, dim=-1)
        h = torch.matmul(w, v)
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)
        return x + h


class InConv(nn.Module):
    """
    First layer conv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class OutConv(nn.Module):
    """
    Last layer conv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Efficient U-Net of LiDAR4D for Ray-drop Refinement
    """
    def __init__(self, in_channels, channels=32, out_channels=1):
        super(UNet, self).__init__()
        self.inc = InConv(in_channels, channels)
        self.down1 = Down(channels, channels*2)     # Down(32, 64)
        self.down2 = Down(channels*2, channels*4)   # Down(64, 128)
        self.down3 = Down(channels*4, channels*8)   # Down(128, 256)
        self.down4 = Down(channels*8, channels*8)   # Down(256, 256)
        self.attn = AttnBlock(channels*8)
        self.up1 = Up(channels*16, channels*4)      # Up(512, 128)
        self.up2 = Up(channels*8, channels*2)       # Up(256, 64)
        self.up3 = Up(channels*4, channels)         # Up(128, 32)
        self.up4 = Up(channels*2, channels)         # Up(64, 32)
        self.outc = OutConv(channels, out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x0 = self.inc(input)                        # 3 -> 32
        x1 = self.down1(x0)                         # 32 -> 64
        x2 = self.down2(x1)                         # 64 -> 128
        x3 = self.down3(x2)                         # 128 -> 256
        x4 = self.down4(x3)                         # 256 -> 256
        x4 = self.attn(x4)
        x_out = self.up1(x4, x3)                    # 256+256 -> 128
        x_out = self.up2(x_out, x2)                 # 128+128 -> 64
        x_out = self.up3(x_out, x1)                 # 64+64 -> 32
        x_out = self.up4(x_out, x0)                 # 32+32 -> 32
        logits = self.outc(x_out)                   # 32 -> 1
        outputs = self.sigmoid(logits)              
        return outputs