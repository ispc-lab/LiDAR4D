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
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, channels=64, num_classes=1):
        super(UNet, self).__init__()
        self.inc = (DoubleConv(in_channels, channels))
        self.down1 = Down(channels, channels*2)     # Down(64, 128)
        self.down2 = Down(channels*2, channels*4)   # Down(128, 256)
        self.down3 = Down(channels*4, channels*8)   # Down(256, 512)
        self.down4 = Down(channels*8, channels*8)   # Down(512, 512)
        self.up1 = Up(channels*16, channels*4)      # Up(1024, 256)
        self.up2 = Up(channels*8, channels*2)       # Up(512, 128)
        self.up3 = Up(channels*4, channels)         # Up(256, 64)
        self.up4 = Up(channels*2, channels)         # Up(128, 64)
        self.outc = (OutConv(channels, num_classes))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x0 = self.inc(input)                        # 3 -> 64
        x1 = self.down1(x0)                         # 64 -> 128
        x2 = self.down2(x1)                         # 128 -> 256
        x3 = self.down3(x2)                         # 256 -> 512
        x4 = self.down4(x3)                         # 512 -> 512
        x_out = self.up1(x4, x3)                    # 512+512 -> 256
        x_out = self.up2(x_out, x2)                 # 256+256 -> 128
        x_out = self.up3(x_out, x1)                 # 128+128 -> 64
        x_out = self.up4(x_out, x0)                 # 128+128 -> 64
        logits = self.outc(x_out)                   # 64 -> 1
        outputs = self.sigmoid(logits)              
        return outputs