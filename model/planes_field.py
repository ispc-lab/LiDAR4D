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
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable
import itertools
import numpy as np


def init_grid_param(
        grid_nd: int,
        in_dim: int,
        out_dim: int,
        reso: Sequence[int],
        a: float = 0.1,
        b: float = 0.5):
    assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
    has_time_planes = in_dim == 4
    assert grid_nd <= in_dim
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
    grid_coefs = nn.ParameterList()
    for ci, coo_comb in enumerate(coo_combs):
        new_grid_coef = nn.Parameter(torch.empty(
            [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
        ))
        if has_time_planes and 3 in coo_comb:  # Initialize time planes to 1
            nn.init.ones_(new_grid_coef)
        else:
            nn.init.uniform_(new_grid_coef, a=a, b=b)
        grid_coefs.append(new_grid_coef)
    return grid_coefs


def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    grid_dim = coords.shape[-1]

    if grid.dim() == grid_dim + 1:
        # no batch dimension present, need to add it
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    if grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample
    else:
        raise NotImplementedError(f"Grid-sample was called with {grid_dim}D data but is only "
                                  f"implemented for 2 and 3D data.")

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]

    # grid_sample expects values in the range of [-1, 1]
    coords_normalized = coords * 2.0 - 1
    interp = grid_sampler(
        grid,  # [B, feature_dim, reso, ...]
        coords_normalized,  # [B, 1, ..., n, grid_dim]
        align_corners=align_corners,
        mode='bilinear', padding_mode='border')
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  # [B, n, feature_dim]
    interp = interp.squeeze()  # [B?, n, feature_dim?]
    return interp


def interpolate_ms_features(pts: torch.Tensor,
                            ms_grids: Collection[Iterable[nn.Module]],
                            grid_dimensions: int,
                            concat_features: bool,
                            num_levels: Optional[int] = None,
                            sample_only = None,
                            ) -> torch.Tensor:
    coo_combs = list(itertools.combinations(
        range(pts.shape[-1]), grid_dimensions)
    )
    if num_levels is None:
        num_levels = len(ms_grids)
    multi_scale_interp_static = [] if concat_features else 0.
    multi_scale_interp_dynamic = [] if concat_features else 0.
    grid: nn.ParameterList
    for scale_id, grid in enumerate(ms_grids[:num_levels]):
        interp_space_static = 1.
        interp_space_dynamic = 1.
        for ci, coo_comb in enumerate(coo_combs):
            if sample_only == 'static' and 3 in coo_comb:
                continue
            if sample_only == 'dynamic' and 3 not in coo_comb:
                continue
            # interpolate in plane
            feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
            interp_out_plane = (
                grid_sample_wrapper(grid[ci], pts[..., coo_comb])
                .view(-1, feature_dim)
            )
            # compute product over planes
            if 3 in coo_comb: # time planes
                interp_space_dynamic = interp_space_dynamic * interp_out_plane
            else:
                interp_space_static = interp_space_static * interp_out_plane

        # combine over scales
        if concat_features:
            multi_scale_interp_static.append(interp_space_static)
            multi_scale_interp_dynamic.append(interp_space_dynamic)
        else:
            multi_scale_interp_static = multi_scale_interp_static + interp_space_static
            multi_scale_interp_dynamic = multi_scale_interp_dynamic + interp_space_dynamic

    if sample_only == 'static':
        if concat_features:
            multi_scale_interp_static = torch.cat(multi_scale_interp_static, dim=-1)
        return multi_scale_interp_static

    if sample_only == 'dynamic':
        if concat_features:
            multi_scale_interp_dynamic = torch.cat(multi_scale_interp_dynamic, dim=-1)
        return multi_scale_interp_dynamic
    
    if concat_features:
        multi_scale_interp_static = torch.cat(multi_scale_interp_static, dim=-1)
        multi_scale_interp_dynamic = torch.cat(multi_scale_interp_dynamic, dim=-1)

    return multi_scale_interp_static, multi_scale_interp_dynamic


class Planes4D(nn.Module):
    def __init__(
        self,
        grid_dimensions=2,
        input_dim=4,
        output_dim=8,
        resolution=[64, 64, 64, 25],
        multiscale_res=[1, 2, 4, 8],
        concat_features=True,
        decompose=True
        ):
        super().__init__()

        self.config = {
            'grid_dimensions': grid_dimensions,
            'input_dim': input_dim,
            'output_dim': output_dim,
            'resolution': resolution,
        }
        self.multiscale_res = multiscale_res
        self.concat_features = concat_features
        self.decompose = decompose

        self.planes = nn.ModuleList()
        self.n_output_dims=0
        for res in self.multiscale_res:
            # initialize coordinate grid
            config = self.config.copy()
            # Resolution fix: multi-res only on spatial planes
            config["resolution"] = [
                r * res for r in config["resolution"][:3]
            ] + config["resolution"][3:]

            gp = init_grid_param(
                grid_nd=config["grid_dimensions"],
                in_dim=config["input_dim"],
                out_dim=config["output_dim"],
                reso=config["resolution"],
            )
            # shape[1] is out-dim - Concatenate over feature len for each scale
            if self.concat_features:
                self.n_output_dims += gp[-1].shape[1]
            else:
                self.n_output_dims = gp[-1].shape[1]
            self.planes.append(gp)
        # print(f"Initialized model grids: {self.planes}")
        if decompose:
            self.n_output_dims *= 2
    
    def forward_static(self, input):
        ## input: [x,y,z,t] shape:[N, 4]
        plane_feat_static = interpolate_ms_features(
            input,
            ms_grids=self.planes,
            grid_dimensions=self.config["grid_dimensions"],
            concat_features=self.concat_features,
            sample_only='static',
            )
        
        return plane_feat_static

    def forward_dynamic(self, input):
        ## input: [x,y,z,t] shape:[N, 4]
        plane_feat_dynamic = interpolate_ms_features(
            input,
            ms_grids=self.planes,
            grid_dimensions=self.config["grid_dimensions"],
            concat_features=self.concat_features,
            sample_only='dynamic',
            )
        
        return plane_feat_dynamic

    def forward(self, input):
        ## input: [x,y,z,t] shape:[N, 4]

        plane_feat_static, plane_feat_dynamic = interpolate_ms_features(
            input,
            ms_grids=self.planes,
            grid_dimensions=self.config["grid_dimensions"],
            concat_features=self.concat_features,
            )
        
        if self.decompose:
            plane_feat = [plane_feat_static, plane_feat_dynamic]
        else:
            plane_feat = plane_feat_static * plane_feat_dynamic

        return plane_feat


if __name__ == '__main__':
    encoder = Planes4D()
    x = torch.rand(100, 4)
    feat_s, feat_d = encoder(x)
    print(feat_s.shape)
    print(feat_d.shape)