# ==============================================================================
# Copyright (c) 2024 Zehan Zheng. All Rights Reserved.
# LiDAR4D: Dynamic Neural Fields for Novel Space-time View LiDAR Synthesis
# CVPR 2024
# https://github.com/ispc-lab/LiDAR4D
# Apache License 2.0
# ==============================================================================

import torch
import torch.nn as nn
import math
import numpy as np
import tinycudann as tcnn


def reduction_func(reduction):
    # x: list of Tensor
    if reduction == 'prod':
        return math.prod
    elif reduction == 'sum':
        return sum
    elif reduction == 'mean':
        return lambda x: sum(x)/len(x)
    elif reduction == 'concat':
        return lambda x: torch.cat(x, dim=-1)
    else:
        raise ValueError("Invalid reduction")


class HashGridT(nn.Module):
    def __init__(
        self,
        time_resolution=8,
        base_resolution=512,
        max_resolution=32768,
        n_levels=8,
        n_features_per_level=4,
        log2_hashmap_size=14,
        num_basis=4,
        ):
        super().__init__()
        self.time_resolution = time_resolution
        per_level_scale = np.exp2(np.log2(max_resolution / base_resolution) / (n_levels - 1))
        hash_t = []
        for _ in range(time_resolution):
            hash_t.append(
                tcnn.Encoding(
                    n_input_dims=2,
                    encoding_config={
                        "otype": "HashGrid",
                        "n_levels": n_levels,
                        "n_features_per_level": n_features_per_level,
                        "log2_hashmap_size": log2_hashmap_size,
                        "base_resolution": base_resolution,
                        "per_level_scale": per_level_scale,
                    },
                )
            )
        self.hash_t = nn.ModuleList(hash_t)
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.num_basis = num_basis
        self.n_output_dims = n_levels * n_features_per_level // num_basis

    def interpT(self, x, t):
        # temporal interpolation in feature dimension
        x = x.view(-1, self.n_levels, self.n_features_per_level)
        num_basis = self.num_basis
        x = torch.chunk(x, num_basis, dim=-1)
        T = [i/(num_basis-1) for i in range(num_basis)]
        basis = lambda j: [(t - T[m]) / (T[j] - T[m]) for m in range(num_basis) if m != j]
        x = sum([math.prod(basis(i)) * x[i] for i in range(num_basis)])
        x = x.view(-1, self.n_output_dims)
        return x

    def forward(self, x, t):
        ## x.shape: [N, 2]
        ## t: float in [0, 1]
        idx = t * (self.time_resolution - 1)
        idx1 = torch.floor(idx).int()
        idx2 = torch.ceil(idx).int()
        if idx1 == idx2:
            hash_feat = self.hash_t[idx1](x)
        else:
            hash_feat = (idx2-idx) * self.hash_t[idx1](x) + (idx-idx1) * self.hash_t[idx2](x)
        hash_feat = self.interpT(hash_feat, t)

        return hash_feat


class HashGrid4D(nn.Module):
    def __init__(
        self,
        base_resolution=512,
        max_resolution=32768,
        time_resolution=8,
        n_levels=8,
        n_features_per_level=4,
        log2_hashmap_size=19,
        hash_size_dynamic=[15, 13, 13], # larger for xy
        decompose=True,     # static & dynamic
        reduction='concat', # 'concat'/'prod'/'sum'/'mean'
        ):
        super().__init__()
        # xyz
        per_level_scale = np.exp2(np.log2(max_resolution / base_resolution) / (n_levels - 1))
        self.hash_static = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": n_features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
            },
        )

        # xyt, xzt, yzt
        hash_dynamic = []
        for i in range(3):
            hash_dynamic.append(
                HashGridT(
                    time_resolution=time_resolution,
                    base_resolution=base_resolution,
                    max_resolution=max_resolution,
                    n_levels=n_levels,
                    n_features_per_level=n_features_per_level,
                    log2_hashmap_size=hash_size_dynamic[i],
                  )
              )
        self.hash_dynamic = nn.ModuleList(hash_dynamic)

        self.decompose = decompose
        self.reduction = reduction
        if reduction == 'concat':
            self.n_output_dims = self.hash_static.n_output_dims + self.hash_dynamic[0].n_output_dims * 3
        else:
            self.n_output_dims = self.hash_static.n_output_dims + self.hash_dynamic[0].n_output_dims

    def forward_static(self, x):
        hash_feat_static = self.hash_static(x) # xyz grid

        return hash_feat_static

    def forward_dynamic(self, x, t):
        xy = x[:, [0,1]]
        xz = x[:, [0,2]]
        yz = x[:, [1,2]]
        
        hash_feat_xyt = self.hash_dynamic[0](xy, t=t)  # xyt grid
        hash_feat_xzt = self.hash_dynamic[1](xz, t=t)  # xzt grid
        hash_feat_yzt = self.hash_dynamic[2](yz, t=t)  # yzt grid

        reduction = reduction_func(self.reduction)
        hash_feat_dynamic = reduction([hash_feat_xyt, hash_feat_xzt, hash_feat_yzt])

        return hash_feat_dynamic

    def forward(self, x, t):
        ## x.shape: [N, 3], in [0, 1]
        ## t: float, in [0, 1]
        hash_feat_static = self.forward_static(x)

        hash_feat_dynamic = self.forward_dynamic(x, t)

        if self.decompose:
            hash_feat = [hash_feat_static, hash_feat_dynamic]
        else:
            hash_feat = torch.cat([hash_feat_static, hash_feat_dynamic], dim=-1)

        return hash_feat


if __name__ == '__main__':
    encoder = HashGrid4D()
    x = torch.rand(100, 3).cuda()
    t = torch.tensor(0.2).cuda()
    feat_s, feat_d = encoder(x, t)
    print(feat_s.shape)
    print(feat_d.shape)