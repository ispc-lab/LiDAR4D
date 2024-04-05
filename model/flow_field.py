# ==============================================================================
# Copyright (c) 2024 Zehan Zheng. All Rights Reserved.
# LiDAR4D: Dynamic Neural Fields for Novel Space-time View LiDAR Synthesis
# CVPR 2024
# https://github.com/ispc-lab/LiDAR4D
# Apache License 2.0
# ==============================================================================

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class FreqEmbed(nn.Module):
    def __init__(
            self,
            num_freqs, 
            linspace=True,
        ):
        super().__init__()
        self.embed_functions = [torch.sin, torch.cos]
        if linspace:
            self.freqs = torch.linspace(1, num_freqs + 1, steps=num_freqs)
        else:
            exps = torch.linspace(0, num_freqs - 1, steps=num_freqs)
            self.freqs = 2**exps

    def forward(self, x):
        embed = []
        for f in self.embed_functions:
            for freq in self.freqs:
                embed.append(f(freq * x * torch.pi))
        embed = torch.cat(embed, -1)

        return embed


class FlowField(nn.Module):
    def __init__(
            self,
            input_dim=4,
            num_layers=8,
            hidden_dim=128,
            num_freqs=6,
        ):
        super().__init__()

        self.pos_enc = FreqEmbed(num_freqs=num_freqs)
        
        layer_list = []
        for l in range(num_layers):
            if l == 0:
                in_dim = input_dim * num_freqs * 2 # pos_enc(xt)
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 6 # forward & backward flow
            else:
                out_dim = hidden_dim
            layer_list.append(nn.Linear(in_dim, out_dim, bias=False))
            if l != num_layers - 1:
                layer_list.append(nn.ReLU())
        self.mlp = nn.Sequential(*layer_list)
        # self.mlp[-1].weight.data.fill_(0.0)
        torch.nn.init.normal_(self.mlp[-1].weight.data, 0, 0.001)

    def forward(self, xt):
        xt = self.pos_enc(xt)

        flow = self.mlp(xt)

        return flow



if __name__ == '__main__':
    encoder = FlowField()
    x = torch.rand(100, 4)
    flow = encoder(x)
    print(flow.shape)