import numpy as np
import torch
from packaging import version as pver
from dataclasses import dataclass
# import trimesh


def custom_meshgrid(*args):
    if pver.parse(torch.__version__) < pver.parse("1.10"):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing="ij")


@torch.cuda.amp.autocast(enabled=False)
def get_lidar_rays(poses, intrinsics, H, W, N=-1, patch_size=1):
    """
    Get lidar rays.

    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [2]
        H, W, N: int
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    """
    device = poses.device
    B = poses.shape[0]

    i, j = custom_meshgrid(
        torch.linspace(0, W - 1, W, device=device),
        torch.linspace(0, H - 1, H, device=device),
    )  # float
    i = i.t().reshape([1, H * W]).expand([B, H * W])
    j = j.t().reshape([1, H * W]).expand([B, H * W])
    results = {}
    if N > 0:
        N = min(N, H * W)

        if isinstance(patch_size, int):
            patch_size_x, patch_size_y = patch_size, patch_size
        elif len(patch_size) == 1:
            patch_size_x, patch_size_y = patch_size[0], patch_size[0]
        else:
            patch_size_x, patch_size_y = patch_size

        if patch_size_x > 0:
            # patch-based random sampling (overlapped)
            num_patch = N // (patch_size_x * patch_size_y)
            inds_x = torch.randint(0, H - patch_size_x, size=[num_patch], device=device)
            inds_y = torch.randint(0, W, size=[num_patch], device=device)
            inds = torch.stack([inds_x, inds_y], dim=-1)  # [np, 2]

            # create meshgrid for each patch
            pi, pj = custom_meshgrid(
                torch.arange(patch_size_x, device=device),
                torch.arange(patch_size_y, device=device),
            )
            offsets = torch.stack([pi.reshape(-1), pj.reshape(-1)], dim=-1)  # [p^2, 2]

            inds = inds.unsqueeze(1) + offsets.unsqueeze(0)  # [np, p^2, 2]
            inds = inds.view(-1, 2)  # [N, 2]
            inds[:, 1] = inds[:, 1] % W
            inds = inds[:, 0] * W + inds[:, 1]  # [N], flatten

            inds = inds.expand([B, N])

        else:
            inds = torch.randint(0, H * W, size=[N], device=device)  # may duplicate
            inds = inds.expand([B, N])

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results["inds"] = inds

    else:
        inds = torch.arange(H * W, device=device).expand([B, H * W])
        results["inds"] = inds

    fov_up, fov = intrinsics
    beta = -(i - W / 2) / W * 2 * np.pi
    alpha = (fov_up - j / H * fov) / 180 * np.pi

    directions = torch.stack(
        [
            torch.cos(alpha) * torch.cos(beta),
            torch.cos(alpha) * torch.sin(beta),
            torch.sin(alpha),
        ],
        -1,
    )
    # directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2)  # (B, N, 3)
    rays_o = poses[..., :3, 3]  # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d)  # [B, N, 3]

    results["rays_o"] = rays_o
    results["rays_d"] = rays_d

    return results


# def visualize_poses(poses, size=0.1):
#     # poses: [B, 4, 4]

#     axes = trimesh.creation.axis(axis_length=4)
#     box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
#     box.colors = np.array([[128, 128, 128]] * len(box.entities))
#     objects = [axes, box]

#     for pose in poses:
#         # a camera is visualized with 8 line segments.
#         pos = pose[:3, 3]
#         a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
#         b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
#         c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
#         d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

#         dir = (a + b + c + d) / 4 - pos
#         dir = dir / (np.linalg.norm(dir) + 1e-8)
#         o = pos + dir * 3

#         segs = np.array(
#             [
#                 [pos, a],
#                 [pos, b],
#                 [pos, c],
#                 [pos, d],
#                 [a, b],
#                 [b, c],
#                 [c, d],
#                 [d, a],
#                 [pos, o],
#             ]
#         )
#         segs = trimesh.load_path(segs)
#         objects.append(segs)

#     trimesh.Scene(objects).show()


@dataclass
class BaseDataset:
    pass
