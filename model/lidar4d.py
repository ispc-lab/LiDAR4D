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
import tinycudann as tcnn
from model.activation import trunc_exp
from model.renderer import LiDAR_Renderer
from model.planes_field import Planes4D
from model.hash_field import HashGrid4D
from model.flow_field import FlowField
from model.unet import UNet


class LiDAR4D(LiDAR_Renderer):
    def __init__(
        self,
        min_resolution=32,
        base_resolution=512,
        max_resolution=32768,
        time_resolution=8,
        n_levels_plane=4,
        n_features_per_level_plane=8,
        n_levels_hash=8,
        n_features_per_level_hash=4,
        log2_hashmap_size=19,
        num_layers_flow=3,
        hidden_dim_flow=64,
        num_layers_sigma=2,
        hidden_dim_sigma=64,
        geo_feat_dim=15,
        num_layers_lidar=3,
        hidden_dim_lidar=64,
        out_lidar_dim=2,
        num_frames=51,
        bound=1,
        **kwargs,
    ):
        super().__init__(bound, **kwargs)

        self.out_lidar_dim = out_lidar_dim
        self.num_frames = num_frames

        self.planes_encoder = Planes4D(
            grid_dimensions=2,
            input_dim=4,
            output_dim=n_features_per_level_plane,
            resolution=[min_resolution] * 3 + [time_resolution],
            multiscale_res=[2**(n) for n in range(n_levels_plane)],
        )

        self.hash_encoder = HashGrid4D(
            base_resolution=base_resolution,
            max_resolution=max_resolution,
            time_resolution=time_resolution,
            n_levels=n_levels_hash,
            n_features_per_level=n_features_per_level_hash,
            log2_hashmap_size=log2_hashmap_size,
        )

        self.view_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "Frequency",
                "degree": 12,
            },
        )

        self.flow_net = FlowField(
            input_dim=4,
            num_layers=num_layers_flow,
            hidden_dim=hidden_dim_flow,
            use_grid=True,
        )

        self.sigma_net = tcnn.Network(
            n_input_dims=self.planes_encoder.n_output_dims + self.hash_encoder.n_output_dims,
            n_output_dims=1 + geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim_sigma,
                "n_hidden_layers": num_layers_sigma - 1,
            },
        )

        self.intensity_net = tcnn.Network(
            n_input_dims=self.view_encoder.n_output_dims + geo_feat_dim,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim_lidar,
                "n_hidden_layers": num_layers_lidar - 1,
            },
        )

        self.raydrop_net = tcnn.Network(
            n_input_dims=self.view_encoder.n_output_dims + geo_feat_dim,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim_lidar,
                "n_hidden_layers": num_layers_lidar - 1,
            },
        )

        self.unet = UNet(in_channels=3, out_channels=1)

    def forward(self, x, d, t):
        pass

    def flow(self, x, t):
        # x: [N, 3] in [-bound, bound] for point clouds
        x = (x + self.bound) / (2 * self.bound)

        if t.shape[0] == 1:
            t = t.repeat(x.shape[0], 1)
        xt = torch.cat([x, t], dim=-1)

        flow = self.flow_net(xt)

        return {
            "forward": flow[:, :3],
            "backward": flow[:, 3:],
        }

    def density(self, x, t=None):
        # x: [N, 3], in [-bound, bound]
        x = (x + self.bound) / (2 * self.bound)  # to [0, 1]

        frame_idx = int(t * (self.num_frames - 1))

        hash_feat_s, hash_feat_d = self.hash_encoder(x, t)

        if t.shape[0] == 1:
            t = t.repeat(x.shape[0], 1)
        xt = torch.cat([x, t], dim=-1)

        plane_feat_s, plane_feat_d = self.planes_encoder(xt)

        # integrate neighboring dynamic features
        flow = self.flow_net(xt)
        hash_feat_1 = hash_feat_2 = hash_feat_d
        plane_feat_1 = plane_feat_2 = plane_feat_d
        if frame_idx < self.num_frames - 1:
            x1 = x + flow[:, :3]
            t1 = torch.tensor((frame_idx + 1) / self.num_frames)
            with torch.no_grad():
                hash_feat_1 = self.hash_encoder.forward_dynamic(x1, t1)
            t1 = t1.repeat(x1.shape[0], 1).to(x1.device)
            xt1 = torch.cat([x1, t1], dim=-1)
            plane_feat_1 = self.planes_encoder.forward_dynamic(xt1)

        if frame_idx > 0:
            x2 = x + flow[:, 3:]
            t2 = torch.tensor((frame_idx - 1) / self.num_frames)
            with torch.no_grad():
                hash_feat_2 = self.hash_encoder.forward_dynamic(x2, t2)
            t2 = t2.repeat(x2.shape[0], 1).to(x2.device)
            xt2 = torch.cat([x2, t2], dim=-1)
            plane_feat_2 = self.planes_encoder.forward_dynamic(xt2)

        plane_feat_d = 0.5 * plane_feat_d + 0.25 * (plane_feat_1 + plane_feat_2)
        hash_feat_d = 0.5 * hash_feat_d + 0.25 * (hash_feat_1 + hash_feat_2)

        features = torch.cat([plane_feat_s, plane_feat_d,
                              hash_feat_s, hash_feat_d], dim=-1)

        h = self.sigma_net(features)
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            "sigma": sigma,
            "geo_feat": geo_feat,
        }

    # allow masked inference
    def attribute(self, x, d, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool
        x = (x + self.bound) / (2 * self.bound)  # to [0, 1]

        if mask is not None:
            output = torch.zeros(
                mask.shape[0], self.out_lidar_dim, dtype=x.dtype, device=x.device
            )  # [N, 3]
            # in case of empty mask
            if not mask.any():
                return output
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        d = (d + 1) / 2  # to [0, 1]
        d = self.view_encoder(d)

        intensity = self.intensity_net(torch.cat([d, geo_feat], dim=-1))
        intensity = torch.sigmoid(intensity)

        raydrop = self.raydrop_net(torch.cat([d, geo_feat], dim=-1))
        raydrop = torch.sigmoid(raydrop)

        h = torch.cat([raydrop, intensity], dim=-1)

        if mask is not None:
            output[mask] = h.to(output.dtype)  # fp16 --> fp32
        else:
            output = h

        return output

    # optimizer utils
    def get_params(self, lr):
        params = [
            {"params": self.planes_encoder.parameters(), "lr": lr},
            {"params": self.hash_encoder.parameters(), "lr": lr},
            {"params": self.view_encoder.parameters(), "lr": lr},
            {"params": self.flow_net.parameters(), "lr": 0.1 * lr},       
            {"params": self.sigma_net.parameters(), "lr": 0.1 * lr},
            {"params": self.intensity_net.parameters(), "lr": 0.1 * lr},
            {"params": self.raydrop_net.parameters(), "lr": 0.1 * lr},
        ]

        return params


if __name__ == '__main__':
    model = LiDAR4D().cuda()
    x = torch.rand(100, 3).cuda()
    t = torch.tensor([0.2]).cuda()
    result = model.density(x, t)
    print(result)