# ==============================================================================
# Copyright (c) 2024 Zehan Zheng. All Rights Reserved.
# LiDAR4D: Dynamic Neural Fields for Novel Space-time View LiDAR Synthesis
# CVPR 2024
# https://github.com/ispc-lab/LiDAR4D
# Apache License 2.0
# ==============================================================================

import torch
import torch.nn as nn


class LiDAR_Renderer(nn.Module):
    def __init__(
        self,
        bound=1,
        near_lidar=0.01,
        far_lidar=0.81,
        density_scale=1,
        active_sensor=False,
    ):
        super().__init__()

        self.bound = bound
        self.near_lidar = near_lidar
        self.far_lidar = far_lidar
        self.density_scale = density_scale
        self.active_sensor = active_sensor

        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        aabb = torch.FloatTensor([-bound, -bound, -bound, bound, bound, bound])
        self.register_buffer("aabb", aabb)

    def forward(self, x, d):
        raise NotImplementedError()

    # separated density and intensity/raydrop query (can accelerate non-cuda-ray mode.)
    def density(self, x):
        raise NotImplementedError()

    def attribute(self, x, d, mask=None, **kwargs):
        raise NotImplementedError()

    def run(
        self,
        rays_o,
        rays_d,
        time, 
        num_steps=768,
        perturb=False,
        **kwargs
    ):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # time: [B, 1]
        # return: image: [B, N, 3], depth: [B, N]

        out_lidar_dim = self.out_lidar_dim

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0]  # N = B * N, in fact
        device = rays_o.device

        aabb = self.aabb

        # hard code
        nears = torch.ones(N, dtype=rays_o.dtype, device=rays_o.device) * self.near_lidar
        fars = torch.ones(N, dtype=rays_o.dtype, device=rays_o.device) * self.far_lidar

        nears.unsqueeze_(-1)
        fars.unsqueeze_(-1)

        # print(f'nears = {nears.min().item()} ~ {nears.max().item()}, fars = {fars.min().item()} ~ {fars.max().item()}')

        z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0)  # [1, T]
        z_vals = z_vals.expand((N, num_steps))  # [N, T]
        z_vals = nears + (fars - nears) * z_vals  # [N, T], in [nears, fars]

        # perturb z_vals
        sample_dist = (fars - nears) / num_steps
        if perturb:
            z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist
            # z_vals = z_vals.clamp(nears, fars) # avoid out of bounds xyzs.

        # generate xyzs
        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1)  # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:])  # a manual clip.

        # query SDF and RGB
        density_outputs = self.density(xyzs.reshape(-1, 3), time)

        # sigmas = density_outputs['sigma'].view(N, num_steps) # [N, T]
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(N, num_steps, -1)

        deltas = z_vals[..., 1:] - z_vals[..., :-1]  # [N, T+t-1]
        deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
        alphas = 1 - torch.exp(-deltas * self.density_scale * density_outputs["sigma"].squeeze(-1))  # [N, T+t]
        if self.active_sensor:
            alphas = 1 - torch.exp(-2 * deltas * self.density_scale * density_outputs["sigma"].squeeze(-1))  # [N, T+t]
        alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1)  # [N, T+t+1]
        weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1]  # [N, T+t]

        dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(-1, v.shape[-1])

        mask = weights > 1e-4  # hard coded
        attr = self.attribute(
            xyzs.reshape(-1, 3),
            dirs.reshape(-1, 3),
            mask=mask.reshape(-1),
            **density_outputs
        )

        attr = attr.view(N, -1, out_lidar_dim)  # [N, T+t, 3]

        # calculate weight_sum (mask)
        weights_sum = weights.sum(dim=-1)  # [N]

        # calculate depth  Note: not real depth!!
        # ori_z_vals = ((z_vals - nears) / (fars - nears)).clamp(0, 1)
        # depth = torch.sum(weights * ori_z_vals, dim=-1)
        depth = torch.sum(weights * z_vals, dim=-1)

        # calculate lidar attributes
        image = torch.sum(weights.unsqueeze(-1) * attr, dim=-2)  # [N, 3], in [0, 1]

        image = image.view(*prefix, out_lidar_dim)
        depth = depth.view(*prefix)

        return {
            "depth_lidar": depth,
            "image_lidar": image,
            "weights_sum_lidar": weights_sum,
            "weights": weights,
            "z_vals": z_vals,
        }

    def render(
        self,
        rays_o,
        rays_d,
        time, 
        staged=False,
        max_ray_batch=4096,
        **kwargs
    ):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: pred_rgb: [B, N, 3]

        _run = self.run

        B, N = rays_o.shape[:2]
        device = rays_o.device

        if staged:
            out_lidar_dim = self.out_lidar_dim
            res_keys = ["depth_lidar", "image_lidar"]
            depth = torch.empty((B, N), device=device)
            image = torch.empty((B, N, out_lidar_dim), device=device)

            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)
                    results_ = _run(
                        rays_o[b : b + 1, head:tail],
                        rays_d[b : b + 1, head:tail],
                        time[b:b+1], 
                        **kwargs
                    )
                    depth[b : b + 1, head:tail] = results_[res_keys[0]]
                    image[b : b + 1, head:tail] = results_[res_keys[1]]
                    head += max_ray_batch

            results = {}
            results[res_keys[0]] = depth
            results[res_keys[1]] = image

        else:
            results = _run(rays_o, rays_d, time, **kwargs)

        return results
