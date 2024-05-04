# ==============================================================================
# Copyright (c) 2024 Zehan Zheng. All Rights Reserved.
# LiDAR4D: Dynamic Neural Fields for Novel Space-time View LiDAR Synthesis
# CVPR 2024
# https://github.com/ispc-lab/LiDAR4D
# Apache License 2.0
# ==============================================================================

import os
import torch
import numpy as np
import configargparse

from model.lidar4d import LiDAR4D
from model.runner import Trainer
from utils.metrics import DepthMeter, IntensityMeter, RaydropMeter, PointsMeter
from utils.misc import set_seed


def get_arg_parser():
    parser = configargparse.ArgumentParser()

    parser.add_argument("--config", is_config_file=True, default="configs/kitti360_4950.txt", help="config file path")
    parser.add_argument("--workspace", type=str, default="workspace")
    parser.add_argument("--refine", action="store_true", help="refine mode")
    parser.add_argument("--test", action="store_true", help="test mode")
    parser.add_argument("--test_eval", action="store_true", help="test and eval mode")
    parser.add_argument("--seed", type=int, default=0)

    ### dataset
    parser.add_argument("--dataloader", type=str, choices=("kitti360", "nuscenes"), default="kitti360")
    parser.add_argument("--path", type=str, default="data/kitti360", help="dataset root path")
    parser.add_argument("--sequence_id", type=str, default="4950")
    parser.add_argument("--preload", type=bool, default=True, help="preload all data into GPU, accelerate training but use more GPU memory")
    parser.add_argument("--bound", type=float, default=1, help="assume the scene is bounded in box[-bound, bound]^3")
    parser.add_argument("--scale", type=float, default=0.01, help="scale lidar location into box[-bound, bound]^3")
    parser.add_argument("--offset", type=float, nargs="*", default=[0, 0, 0], help="offset of lidar location")
    parser.add_argument("--near_lidar", type=float, default=1.0, help="minimum near distance for lidar")
    parser.add_argument("--far_lidar", type=float, default=81.0, help="maximum far distance for lidar")
    parser.add_argument("--fov_lidar", type=float, nargs="*", default=[2.0, 26.9], help="fov up and fov range of lidar")
    parser.add_argument("--num_frames", type=int, default=51, help="total number of sequence frames")

    ### LiDAR4D
    parser.add_argument("--min_resolution", type=int, default=32, help="minimum resolution for planes")
    parser.add_argument("--base_resolution", type=int, default=512, help="minimum resolution for hash grid")
    parser.add_argument("--max_resolution", type=int, default=32768, help="maximum resolution for hash grid")
    parser.add_argument("--time_resolution", type=int, default=8, help="temporal resolution")
    parser.add_argument("--n_levels_plane", type=int, default=4, help="n_levels for planes")
    parser.add_argument("--n_features_per_level_plane", type=int, default=8, help="n_features_per_level for planes")
    parser.add_argument("--n_levels_hash", type=int, default=8, help="n_levels for hash grid")
    parser.add_argument("--n_features_per_level_hash", type=int, default=4, help="n_features_per_level for hash grid")
    parser.add_argument("--log2_hashmap_size", type=int, default=19, help="hashmap size for hash grid")
    parser.add_argument("--num_layers_flow", type=int, default=3, help="num_layers of flownet")
    parser.add_argument("--hidden_dim_flow", type=int, default=64, help="hidden_dim of flownet")
    parser.add_argument("--num_layers_sigma", type=int, default=2, help="num_layers of sigmanet")
    parser.add_argument("--hidden_dim_sigma", type=int, default=64, help="hidden_dim of sigmanet")
    parser.add_argument("--geo_feat_dim", type=int, default=15, help="geo_feat_dim of sigmanet")
    parser.add_argument("--num_layers_lidar", type=int, default=3, help="num_layers of intensity/raydrop")
    parser.add_argument("--hidden_dim_lidar", type=int, default=64, help="hidden_dim of intensity/raydrop")
    parser.add_argument("--out_lidar_dim", type=int, default=2, help="output dim for lidar intensity/raydrop")

    ### training
    parser.add_argument("--depth_loss", type=str, default="l1", help="l1, bce, mse, huber")
    parser.add_argument("--depth_grad_loss", type=str, default="l1", help="l1, bce, mse, huber")
    parser.add_argument("--intensity_loss", type=str, default="mse", help="l1, bce, mse, huber")
    parser.add_argument("--raydrop_loss", type=str, default="mse", help="l1, bce, mse, huber")
    parser.add_argument("--flow_loss", type=bool, default=True)
    parser.add_argument("--grad_loss", type=bool, default=True)

    parser.add_argument("--alpha_d", type=float, default=1)
    parser.add_argument("--alpha_i", type=float, default=0.1)
    parser.add_argument("--alpha_r", type=float, default=0.01)
    parser.add_argument("--alpha_grad", type=float, default=0.1)
    parser.add_argument("--alpha_grad_norm", type=float, default=0.1)
    parser.add_argument("--alpha_spatial", type=float, default=0.1)
    parser.add_argument("--alpha_tv", type=float, default=0.1)

    parser.add_argument("--grad_norm_smooth", action="store_true")
    parser.add_argument("--spatial_smooth", action="store_true")
    parser.add_argument("--tv_loss", action="store_true")
    parser.add_argument("--sobel_grad", action="store_true")
    parser.add_argument("--urf_loss", action="store_true", help="enable line-of-sight loss in URF.")
    parser.add_argument("--active_sensor", action="store_true", help="enable volume rendering for active sensor.")

    parser.add_argument("--density_scale", type=float, default=1)
    parser.add_argument("--intensity_scale", type=float, default=1)
    parser.add_argument("--raydrop_ratio", type=float, default=0.5)
    parser.add_argument("--smooth_factor", type=float, default=0.2)

    parser.add_argument("--iters", type=int, default=30000, help="training iters")
    parser.add_argument("--lr", type=float, default=1e-2, help="initial learning rate")
    parser.add_argument("--fp16", type=bool, default=True, help="use amp mixed precision training")
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--ckpt", type=str, default="latest")
    parser.add_argument("--num_rays_lidar", type=int, default=1024, help="num rays sampled per image for each training step")
    parser.add_argument("--num_steps", type=int, default=768, help="num steps sampled per ray")
    parser.add_argument("--patch_size_lidar", type=int, default=1, help="[experimental] render patches in training." 
                                                                        "1 means disabled, use [64, 32, 16] to enable")
    parser.add_argument("--change_patch_size_lidar", nargs="+", type=int, default=[2, 8], help="[experimental] render patches in training. " 
                                                                      "1 means disabled, use [64, 32, 16] to enable, change during training")
    parser.add_argument("--change_patch_size_epoch", type=int, default=2, help="change patch_size intenvel")
    parser.add_argument("--ema_decay", type=float, default=0.95, help="use ema during training")

    return parser


def main():
    parser = get_arg_parser()
    opt = parser.parse_args()
    set_seed(opt.seed)

    # Check sequence id.
    kitti360_sequence_ids = [
        "1538",
        "1728",
        "1908",
        "3353",
        "2350",
        "4950",
        "8120",
        "10200",
        "10750",
        "11400",
    ]

    # Specify dataloader class
    if opt.dataloader == "kitti360":
        from data.kitti360_dataset import KITTI360Dataset as NeRFDataset

        if opt.sequence_id not in kitti360_sequence_ids:
            raise ValueError(
                f"Unknown sequence id {opt.sequence_id} for {opt.dataloader}"
            )
    # elif opt.dataloader == "nuscenes":
    #     from data.nus_dataset import NusDataset as NeRFDataset
    else:
        raise RuntimeError("Should not reach here.")

    # Logging
    os.makedirs(opt.workspace, exist_ok=True)
    f = os.path.join(opt.workspace, "args.txt")
    with open(f, "w") as file:
        for arg in vars(opt):
            attr = getattr(opt, arg)
            file.write("{} = {}\n".format(arg, attr))

    if opt.patch_size_lidar > 1:
        assert (
            opt.num_rays % (opt.patch_size_lidar**2) == 0
        ), "patch_size ** 2 should be dividable by num_rays."

    opt.near_lidar = opt.near_lidar * opt.scale
    opt.far_lidar = opt.far_lidar * opt.scale

    model = LiDAR4D(
        min_resolution=opt.min_resolution,
        base_resolution=opt.base_resolution,
        max_resolution=opt.max_resolution,
        time_resolution=opt.time_resolution,
        n_levels_plane=opt.n_levels_plane,
        n_features_per_level_plane=opt.n_features_per_level_plane,
        n_levels_hash=opt.n_levels_hash,
        n_features_per_level_hash=opt.n_features_per_level_hash,
        log2_hashmap_size=opt.log2_hashmap_size,
        num_layers_flow=opt.num_layers_flow,
        hidden_dim_flow=opt.hidden_dim_flow,
        num_layers_sigma=opt.num_layers_sigma,
        hidden_dim_sigma=opt.hidden_dim_sigma,
        geo_feat_dim=opt.geo_feat_dim,
        num_layers_lidar=opt.num_layers_lidar,
        hidden_dim_lidar=opt.hidden_dim_lidar,
        out_lidar_dim=opt.out_lidar_dim,
        num_frames=opt.num_frames,
        bound=opt.bound,
        near_lidar=opt.near_lidar,
        far_lidar=opt.far_lidar,
        density_scale=opt.density_scale,
        active_sensor=opt.active_sensor,
    )
    # print(model)
    print(opt)
    
    loss_dict = {
        "mse": torch.nn.MSELoss(reduction="none"),
        "l1": torch.nn.L1Loss(reduction="none"),
        "bce": torch.nn.BCEWithLogitsLoss(reduction="none"),
        "huber": torch.nn.HuberLoss(reduction="none", delta=0.2 * opt.scale),
        "cos": torch.nn.CosineSimilarity(),
    }
    criterion = {
        "depth": loss_dict[opt.depth_loss],
        "raydrop": loss_dict[opt.raydrop_loss],
        "intensity": loss_dict[opt.intensity_loss],
        "grad": loss_dict[opt.depth_grad_loss],
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lidar_metrics = [
        RaydropMeter(ratio=opt.raydrop_ratio),
        IntensityMeter(scale=opt.intensity_scale),
        DepthMeter(scale=opt.scale),
        PointsMeter(scale=opt.scale, intrinsics=opt.fov_lidar),
    ]

    if opt.test or opt.test_eval or opt.refine:
        trainer = Trainer(
            "lidar4d",
            opt,
            model,
            device=device,
            workspace=opt.workspace,
            criterion=criterion,
            fp16=opt.fp16,
            lidar_metrics=lidar_metrics,
            use_checkpoint=opt.ckpt,
        )

        if opt.refine: # optimize raydrop only
            refine_loader = NeRFDataset(
                device=device,
                split="refine",
                root_path=opt.path,
                sequence_id=opt.sequence_id,
                preload=opt.preload,
                scale=opt.scale,
                offset=opt.offset,
                fp16=opt.fp16,
                patch_size_lidar=opt.patch_size_lidar,
                num_rays_lidar=opt.num_rays_lidar,
                fov_lidar=opt.fov_lidar,
            ).dataloader()
            trainer.refine(refine_loader)

        test_loader = NeRFDataset(
            device=device,
            split="test",
            root_path=opt.path,
            sequence_id=opt.sequence_id,
            preload=opt.preload,
            scale=opt.scale,
            offset=opt.offset,
            fp16=opt.fp16,
            patch_size_lidar=opt.patch_size_lidar,
            num_rays_lidar=opt.num_rays_lidar,
            fov_lidar=opt.fov_lidar,
        ).dataloader()

        if test_loader.has_gt and not opt.test:
            trainer.evaluate(test_loader)

        trainer.test(test_loader, write_video=False)

    else:  # full pipeline
        train_loader = NeRFDataset(
            device=device,
            split="train",
            root_path=opt.path,
            sequence_id=opt.sequence_id,
            preload=opt.preload,
            scale=opt.scale,
            offset=opt.offset,
            fp16=opt.fp16,
            patch_size_lidar=opt.patch_size_lidar,
            num_rays_lidar=opt.num_rays_lidar,
            fov_lidar=opt.fov_lidar,
        ).dataloader()

        valid_loader = NeRFDataset(
            device=device,
            split="val",
            root_path=opt.path,
            sequence_id=opt.sequence_id,
            preload=opt.preload,
            scale=opt.scale,
            offset=opt.offset,
            fp16=opt.fp16,
            patch_size_lidar=opt.patch_size_lidar,
            num_rays_lidar=opt.num_rays_lidar,
            fov_lidar=opt.fov_lidar,
        ).dataloader()

        # optimize raydrop
        refine_loader = NeRFDataset(
            device=device,
            split="refine",
            root_path=opt.path,
            sequence_id=opt.sequence_id,
            preload=opt.preload,
            scale=opt.scale,
            offset=opt.offset,
            fp16=opt.fp16,
            patch_size_lidar=opt.patch_size_lidar,
            num_rays_lidar=opt.num_rays_lidar,
            fov_lidar=opt.fov_lidar,
        ).dataloader()

        optimizer = lambda model: torch.optim.Adam(
            model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15
        )

        # decay to 0.1 * init_lr at last iter step
        scheduler = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1)
        )

        trainer = Trainer(
            "lidar4d",
            opt,
            model,
            device=device,
            workspace=opt.workspace,
            criterion=criterion,
            fp16=opt.fp16,
            lidar_metrics=lidar_metrics,
            use_checkpoint=opt.ckpt,
            optimizer=optimizer,
            ema_decay=opt.ema_decay,
            lr_scheduler=scheduler,
            scheduler_update_every_step=True,
            eval_interval=opt.eval_interval,
        )

        max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
        print(f"max_epoch: {max_epoch}")
        trainer.train(train_loader, valid_loader, refine_loader, max_epoch)

        # also test
        test_loader = NeRFDataset(
            device=device,
            split="test",
            root_path=opt.path,
            sequence_id=opt.sequence_id,
            preload=opt.preload,
            scale=opt.scale,
            offset=opt.offset,
            fp16=opt.fp16,
            patch_size_lidar=opt.patch_size_lidar,
            num_rays_lidar=opt.num_rays_lidar,
            fov_lidar=opt.fov_lidar,
        ).dataloader()

        if test_loader.has_gt:
            trainer.evaluate(test_loader)  # evaluate metrics

        trainer.test(test_loader, write_video=False)  # save final results



if __name__ == "__main__":
    main()
