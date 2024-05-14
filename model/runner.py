# ==============================================================================
# Copyright (c) 2024 Zehan Zheng. All Rights Reserved.
# LiDAR4D: Dynamic Neural Fields for Novel Space-time View LiDAR Synthesis
# CVPR 2024
# https://github.com/ispc-lab/LiDAR4D
# Apache License 2.0
# ==============================================================================

import glob
import os
import random
import time

import cv2
import imageio
import numpy as np
import tensorboardX
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from rich.console import Console
from torch_ema import ExponentialMovingAverage

from utils.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from utils.convert import pano_to_lidar
from utils.misc import point_removal


class Trainer(object):
    def __init__(
        self,
        name,   # name of this experiment
        opt,    # extra conf
        model,  # network
        criterion=None,  # loss function, if None, assume inline implementation in train_step
        optimizer=None,  # optimizer
        ema_decay=None,  # if use EMA, set the decay
        lr_scheduler=None,  # scheduler
        lidar_metrics=[], # metrics for evaluation, if None, use val_loss to measure performance.
        device=None,  # device to use, usually setting to None is OK. (auto choose device)
        mute=False,   # whether to mute all print
        fp16=False,   # amp optimize level
        eval_interval=50,  # eval once every $ epoch
        max_keep_ckpt=1,   # max num of saved ckpts in disk
        workspace="workspace",  # workspace to save logs & ckpts
        best_mode="min",   # the smaller/larger result, the better
        use_checkpoint="latest",  # which ckpt to use at init time
        use_tensorboardX=True,    # whether to use tensorboard for logging
        scheduler_update_every_step=False,  # whether to call scheduler.step() after every train step
    ):
        self.name = name
        self.opt = opt
        self.mute = mute
        self.lidar_metrics = lidar_metrics
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = (
            device
            if device is not None
            else torch.device(
                f"cuda:0" if torch.cuda.is_available() else "cpu"
            )
        )
        self.console = Console()

        model.to(self.device)
        self.model = model

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        self.bce_fn = torch.nn.BCELoss()
        self.cham_fn = chamfer_3DDist()

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4)  # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1)  # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [],  # metrics or valid_loss
            "checkpoints": [],  # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, "checkpoints")
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)

        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f"[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}")

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else:  # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)
        
        self.runtime_train = []
        self.runtime_test = []

    def __del__(self):
        if self.log_ptr:
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if not self.mute:
            self.console.print(*args, **kwargs)
        if self.log_ptr:
            print(*args, file=self.log_ptr)
            self.log_ptr.flush()  # write immediately to file

    ### ------------------------------

    def train_step(self, data):
        # Initialize all returned values
        pred_intensity = None
        gt_intensity = None
        pred_depth = None
        gt_depth = None
        loss = 0

        rays_o_lidar = data["rays_o_lidar"]  # [B, N, 3]
        rays_d_lidar = data["rays_d_lidar"]  # [B, N, 3]
        time_lidar = data['time'] # [B, 1]
        images_lidar = data["images_lidar"]  # [B, N, 3]

        gt_raydrop = images_lidar[:, :, 0]
        gt_intensity = images_lidar[:, :, 1] * gt_raydrop
        gt_depth = images_lidar[:, :, 2] * gt_raydrop

        outputs_lidar = self.model.render(
            rays_o_lidar,
            rays_d_lidar,
            time_lidar,
            staged=False,
            perturb=True,
            force_all_rays=False if self.opt.patch_size_lidar == 1 else True,
            **vars(self.opt),
        )

        pred_raydrop = outputs_lidar["image_lidar"][:, :, 0]
        pred_intensity = outputs_lidar["image_lidar"][:, :, 1] * gt_raydrop
        pred_depth = outputs_lidar["depth_lidar"] * gt_raydrop

        if self.opt.raydrop_loss == 'bce':
            pred_raydrop = F.sigmoid(pred_raydrop)

        # label smoothing for ray-drop
        smooth = self.opt.smooth_factor # 0.2
        gt_raydrop_smooth = gt_raydrop.clamp(smooth, 1-smooth)

        lidar_loss = (
            self.opt.alpha_d * self.criterion["depth"](pred_depth, gt_depth)
            + self.opt.alpha_r * self.criterion["raydrop"](pred_raydrop, gt_raydrop_smooth)
            + self.opt.alpha_i * self.criterion["intensity"](pred_intensity, gt_intensity)
        )
        pred_intensity = pred_intensity.unsqueeze(-1)
        gt_intensity = gt_intensity.unsqueeze(-1)

        # main loss
        loss = lidar_loss.sum()

        # additional CD Loss
        pred_lidar = rays_d_lidar * pred_depth.unsqueeze(-1) / self.opt.scale
        gt_lidar = rays_d_lidar * gt_depth.unsqueeze(-1) / self.opt.scale
        dist1, dist2, _, _ = self.cham_fn(pred_lidar, gt_lidar)
        chamfer_loss = (dist1 + dist2).mean() * 0.5
        loss = loss + chamfer_loss

        if self.opt.flow_loss:
            frame_idx = int(time_lidar * (self.opt.num_frames - 1))
            pc = self.pc_list[f"{frame_idx}"]
            pc = torch.from_numpy(pc).cuda().float().contiguous()

            pred_flow = self.model.flow(pc, time_lidar)
            pred_flow_forward = pred_flow["forward"]
            pred_flow_backward = pred_flow["backward"]

            # two-step consistency
            for step in [1, 2]:
                if f"{frame_idx+step}" in self.pc_list.keys():
                    pc_pred = pc + pred_flow_forward * step
                    pc_forward = self.pc_list[f"{frame_idx+step}"]
                    pc_forward = torch.from_numpy(pc_forward).cuda().float().contiguous()
                    dist1, dist2, _, _ = self.cham_fn(pc_pred.unsqueeze(0), pc_forward.unsqueeze(0))
                    chamfer_dist = (dist1.sum() + dist2.sum()) * 0.5
                    loss = loss + chamfer_dist

                if f"{frame_idx-step}" in self.pc_list.keys():
                    pc_pred = pc + pred_flow_backward * step
                    pc_backward = self.pc_list[f"{frame_idx-step}"]
                    pc_backward = torch.from_numpy(pc_backward).cuda().float().contiguous()
                    dist1, dist2, _, _ = self.cham_fn(pc_pred.unsqueeze(0), pc_backward.unsqueeze(0))
                    chamfer_dist = (dist1.sum() + dist2.sum()) * 0.5
                    loss = loss + chamfer_dist

            # regularize flow on the ground
            ground = self.pc_ground_list[f"{frame_idx}"]
            ground = torch.from_numpy(ground).cuda().float().contiguous()
            zero_flow = self.model.flow(ground, torch.rand(1).to(time_lidar))
            loss = loss + 0.001 * (zero_flow["forward"].abs().sum() + zero_flow["backward"].abs().sum())

        # line-of-sight loss
        if self.opt.urf_loss:
            eps = 0.02 * 0.1 ** min(self.global_step / self.opt.iters, 1)
            # gt_depth [B, N]
            weights = outputs_lidar["weights"] # [B*N, T]
            z_vals = outputs_lidar["z_vals"]

            depth_mask = gt_depth.reshape(z_vals.shape[0], 1) > 0.0
            mask_empty = (z_vals < (gt_depth.reshape(z_vals.shape[0], 1) - eps)) | (z_vals > (gt_depth.reshape(z_vals.shape[0], 1) + eps))
            loss_empty = ((mask_empty * weights) ** 2).sum() / depth_mask.sum()

            loss = loss + 0.1 * loss_empty

            mask_near = (z_vals > (gt_depth.reshape(z_vals.shape[0], 1) - eps)) & (z_vals < (gt_depth.reshape(z_vals.shape[0], 1) + eps))
            distance = mask_near * (z_vals - gt_depth.reshape(z_vals.shape[0], 1))
            sigma = eps / 3.
            distr = 1.0 / (sigma * np.sqrt(2 * np.pi)) * torch.exp(-(distance ** 2 / (2 * sigma ** 2)))
            distr /= distr.max()
            distr *= mask_near
            loss_near = ((mask_near * weights - distr) ** 2).sum() / depth_mask.sum()

            loss = loss + 0.1 * loss_near

        # gradient loss
        if isinstance(self.opt.patch_size_lidar, int):
            patch_size_x, patch_size_y = self.opt.patch_size_lidar, self.opt.patch_size_lidar
        elif len(self.opt.patch_size_lidar) == 1:
            patch_size_x, patch_size_y = self.opt.patch_size_lidar[0], self.opt.patch_size_lidar[0]
        else:
            patch_size_x, patch_size_y = self.opt.patch_size_lidar

        if patch_size_x > 1:
            pred_depth = pred_depth.view(-1, patch_size_x, patch_size_y, 1).permute(0, 3, 1, 2).contiguous() / self.opt.scale
            if self.opt.sobel_grad:
                pred_grad_x = F.conv2d(
                    pred_depth,
                    torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                 dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device),
                    padding=1,
                    )
                pred_grad_y = F.conv2d(
                    pred_depth,
                    torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                 dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device),
                    padding=1,
                    )
            else:
                pred_grad_y = torch.abs(pred_depth[:, :, :-1, :] - pred_depth[:, :, 1:, :])
                pred_grad_x = torch.abs(pred_depth[:, :, :, :-1] - pred_depth[:, :, :, 1:])

            dy = torch.abs(pred_grad_y)
            dx = torch.abs(pred_grad_x)

            if self.opt.grad_norm_smooth:
                grad_norm = torch.mean(torch.exp(-dx)) + torch.mean(torch.exp(-dy))
                # print('grad_norm', grad_norm)
                loss = loss + self.opt.alpha_grad_norm * grad_norm

            if self.opt.spatial_smooth:
                spatial_loss = torch.mean(dx**2) + torch.mean(dy**2)
                # print('spatial_loss', spatial_loss)
                loss = loss + self.opt.alpha_spatial * spatial_loss

            if self.opt.tv_loss:
                tv_loss = torch.mean(dx) + torch.mean(dy)
                # print('tv_loss', tv_loss)
                loss = loss + self.opt.alpha_tv * tv_loss

            if self.opt.grad_loss:
                gt_depth = gt_depth.view(-1, patch_size_x, patch_size_y, 1).permute(0, 3, 1, 2).contiguous() / self.opt.scale
                gt_raydrop = gt_raydrop.view(-1, patch_size_x, patch_size_y, 1).permute(0, 3, 1, 2).contiguous()

                # sobel
                if self.opt.sobel_grad:
                    gt_grad_y = F.conv2d(
                        gt_depth,
                        torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                     dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device),
                        padding=1,
                        )

                    gt_grad_x = F.conv2d(
                        gt_depth,
                        torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                     dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device),
                        padding=1,
                        )
                else:
                    gt_grad_y = gt_depth[:, :, :-1, :] - gt_depth[:, :, 1:, :]
                    gt_grad_x = gt_depth[:, :, :, :-1] - gt_depth[:, :, :, 1:]

                grad_clip_x = 0.01
                grad_mask_x = torch.where(torch.abs(gt_grad_x) < grad_clip_x, 1, 0)
                grad_clip_y = 0.01
                grad_mask_y = torch.where(torch.abs(gt_grad_y) < grad_clip_y, 1, 0)
                if self.opt.sobel_grad:
                    mask_dx = gt_raydrop * grad_mask_x
                    mask_dy = gt_raydrop * grad_mask_y
                else:
                    mask_dx = gt_raydrop[:, :, :, :-1] * grad_mask_x
                    mask_dy = gt_raydrop[:, :, :-1, :] * grad_mask_y

                if self.opt.depth_grad_loss == "cos":
                    patch_num = pred_grad_x.shape[0]
                    grad_loss = self.criterion["grad"](
                        (pred_grad_x * mask_dx).reshape(patch_num, -1),
                        (gt_grad_x * mask_dx).reshape(patch_num, -1),
                    )
                    grad_loss = 1 - grad_loss
                else:
                    grad_loss = self.criterion["grad"](
                        pred_grad_x * mask_dx, 
                        gt_grad_x * mask_dx
                    )
                loss = loss + self.opt.alpha_grad * grad_loss.sum()

        return (
            pred_intensity,
            gt_intensity,
            pred_depth,
            gt_depth,
            loss,
        )

    def eval_step(self, data):
        pred_intensity = None
        pred_depth = None
        pred_raydrop = None
        gt_intensity = None
        gt_depth = None
        gt_raydrop = None
        loss = 0

        rays_o_lidar = data["rays_o_lidar"]  # [B, N, 3]
        rays_d_lidar = data["rays_d_lidar"]  # [B, N, 3]
        time_lidar = data['time']
        images_lidar = data["images_lidar"]  # [B, H, W, 3]
        H_lidar, W_lidar = data["H_lidar"], data["W_lidar"]

        gt_raydrop = images_lidar[:, :, :, 0]
        gt_intensity = images_lidar[:, :, :, 1] * gt_raydrop
        gt_depth = images_lidar[:, :, :, 2] * gt_raydrop

        outputs_lidar = self.model.render(
            rays_o_lidar,
            rays_d_lidar,
            time_lidar,
            staged=True,
            perturb=False,
            **vars(self.opt),
        )

        pred_rgb_lidar = outputs_lidar["image_lidar"].reshape(-1, H_lidar, W_lidar, 2)
        pred_raydrop = pred_rgb_lidar[:, :, :, 0]
        pred_intensity = pred_rgb_lidar[:, :, :, 1]
        pred_depth = outputs_lidar["depth_lidar"].reshape(-1, H_lidar, W_lidar)
        if self.opt.raydrop_loss == 'bce':
            pred_raydrop = F.sigmoid(pred_raydrop)
        if self.use_refine:
            pred_raydrop = torch.cat([pred_raydrop, pred_intensity, pred_depth], dim=0).unsqueeze(0)
            pred_raydrop = self.model.unet(pred_raydrop).squeeze(0)
        raydrop_mask = torch.where(pred_raydrop > 0.5, 1, 0)

        lidar_loss = (
            self.opt.alpha_d * self.criterion["depth"](pred_depth * raydrop_mask, gt_depth).mean()
            + self.opt.alpha_r * self.criterion["raydrop"](pred_raydrop, gt_raydrop).mean()
            + self.opt.alpha_i * self.criterion["intensity"](pred_intensity * raydrop_mask, gt_intensity).mean()
        )

        loss = lidar_loss
        
        return (
            pred_intensity,
            pred_depth,
            pred_raydrop,
            gt_intensity,
            gt_depth,
            gt_raydrop,
            loss,
        )

    def test_step(self, data, perturb=False):
        pred_raydrop = None
        pred_intensity = None
        pred_depth = None

        rays_o_lidar = data["rays_o_lidar"]  # [B, N, 3]

        rays_d_lidar = data["rays_d_lidar"]  # [B, N, 3]
        time_lidar = data['time']
        H_lidar, W_lidar = data["H_lidar"], data["W_lidar"]

        outputs_lidar = self.model.render(
            rays_o_lidar,
            rays_d_lidar,
            time_lidar,
            staged=True,
            perturb=perturb,
            **vars(self.opt),
        )

        pred_rgb_lidar = outputs_lidar["image_lidar"].reshape(-1, H_lidar, W_lidar, 2)
        pred_raydrop = pred_rgb_lidar[:, :, :, 0]
        pred_intensity = pred_rgb_lidar[:, :, :, 1]
        pred_depth = outputs_lidar["depth_lidar"].reshape(-1, H_lidar, W_lidar)
        if self.opt.raydrop_loss == 'bce':
            pred_raydrop = F.sigmoid(pred_raydrop)
        if self.use_refine:
            pred_raydrop = torch.cat([pred_raydrop, pred_intensity, pred_depth], dim=0).unsqueeze(0)
            pred_raydrop = self.model.unet(pred_raydrop).squeeze(0)
        raydrop_mask = torch.where(pred_raydrop > 0.5, 1, 0)
        if self.opt.alpha_r > 0:
            pred_intensity = pred_intensity * raydrop_mask
            pred_depth = pred_depth * raydrop_mask

        return pred_raydrop, pred_intensity, pred_depth

    ### ------------------------------

    def train_one_epoch(self, loader):
        log_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.log(
            f"[{log_time}] ==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ..."
        )

        total_loss = 0

        self.model.train()

        pbar = tqdm.tqdm(
            total=len(loader) * loader.batch_size,
            bar_format="{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        self.local_step = 0

        for data in loader:
            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                (
                    pred_intensity,
                    gt_intensity,
                    pred_depth,
                    gt_depth,
                    loss,
                ) = self.train_step(data)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.use_tensorboardX:
                self.writer.add_scalar("train/loss", loss_val, self.global_step)
                self.writer.add_scalar(
                    "train/lr",
                    self.optimizer.param_groups[0]["lr"],
                    self.global_step,
                )

            if self.scheduler_update_every_step:
                pbar.set_description(
                    f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}"
                )
            else:
                pbar.set_description(
                    f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})"
                )
            pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)
        self.log(f"average_loss: {average_loss}.")

        pbar.close()

        if not self.scheduler_update_every_step:
            if isinstance(
                self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")

    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        if name is None:
            name = f"{self.name}_ep{self.epoch:04d}"

        total_loss = 0
        for metric in self.lidar_metrics:
            metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        pbar = tqdm.tqdm(
            total=len(loader) * loader.batch_size,
            bar_format="{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        with torch.no_grad():
            self.local_step = 0

            for data in loader:
                self.local_step += 1

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    (
                        preds_intensity,
                        preds_depth,
                        preds_raydrop,
                        gt_intensity,
                        gt_depth,
                        gt_raydrop,
                        loss,
                    ) = self.eval_step(data)

                preds_mask = torch.where(preds_raydrop > 0.5, 1, 0)

                loss_val = loss.item()
                total_loss += loss_val

                for i, metric in enumerate(self.lidar_metrics):
                    if i == 0:  # hard code
                        metric.update(preds_raydrop, gt_raydrop)
                    elif i == 1:
                        metric.update(preds_intensity*preds_mask, gt_intensity)
                    else:
                        metric.update(preds_depth*preds_mask, gt_depth)

                save_path_pred = os.path.join(
                    self.workspace,
                    "validation",
                    f"{name}_{self.local_step:04d}.png",
                )
                os.makedirs(os.path.dirname(save_path_pred), exist_ok=True)

                pred_raydrop = preds_raydrop[0].detach().cpu().numpy()
                img_raydrop = (pred_raydrop * 255).astype(np.uint8)
                img_raydrop = cv2.cvtColor(img_raydrop, cv2.COLOR_GRAY2BGR)

                pred_intensity = preds_intensity[0].detach().cpu().numpy()
                img_intensity = (pred_intensity * 255).astype(np.uint8)
                img_intensity = cv2.applyColorMap(img_intensity, 1)
                
                pred_depth = preds_depth[0].detach().cpu().numpy()
                img_depth = (pred_depth * 255).astype(np.uint8)
                img_depth = cv2.applyColorMap(img_depth, 20)

                preds_mask = preds_mask[0].detach().cpu().numpy()
                img_mask = (preds_mask * 255).astype(np.uint8)
                img_raydrop_masked = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)

                img_intensity_masked = (pred_intensity * preds_mask * 255).astype(np.uint8)
                img_intensity_masked = cv2.applyColorMap(img_intensity_masked, 1)
                
                img_depth_masked = (pred_depth * preds_mask * 255).astype(np.uint8)
                img_depth_masked = cv2.applyColorMap(img_depth_masked, 20)

                img_pred = cv2.vconcat([img_raydrop, img_intensity, img_depth, 
                                        img_raydrop_masked, img_intensity_masked, img_depth_masked])
                
                cv2.imwrite(save_path_pred, img_pred)
                
                ## save point clouds
                # pred_lidar = pano_to_lidar(
                #     pred_depth / self.opt.scale, loader._data.intrinsics_lidar
                # )
                # np.save(
                #     os.path.join(
                #         self.workspace,
                #         "validation",
                #         f"{name}_{self.local_step:04d}_lidar.npy",
                #     ),
                #     pred_lidar,
                # )

                pbar.set_description(
                    f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})"
                )
                pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        pbar.close()
        if len(self.lidar_metrics) > 0:
            result = self.lidar_metrics[-1].measure()[0]  # hard code
            self.stats["results"].append(
                result if self.best_mode == "min" else -result
            )  # if max mode, use -result
        else:
            self.stats["results"].append(
                average_loss
            )  # if no metric, choose best by min loss

        np.set_printoptions(linewidth=150, suppress=True, precision=8)
        for i, metric in enumerate(self.lidar_metrics):
            if i == 1:
                self.log(f"== ↓ Final pred ↓ == RMSE{' '*6}MedAE{' '*6}LPIPS{' '*8}SSIM{' '*8}PSNR ===")
            self.log(metric.report(), style="blue")
            if self.use_tensorboardX:
                metric.write(self.writer, self.epoch, prefix="LiDAR_evaluate")
            metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    ### ------------------------------

    def train(self, train_loader, valid_loader, refine_loader, max_epochs):
        if self.use_tensorboardX:
            summary_path = os.path.join(self.workspace, "run", self.name)
            self.writer = tensorboardX.SummaryWriter(summary_path)

        if self.opt.flow_loss:
            self.process_pointcloud(refine_loader)

        change_dataloder = False
        if self.opt.change_patch_size_lidar[0] > 1:
            change_dataloder = True
        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch
            if change_dataloder:
                if self.epoch % self.opt.change_patch_size_epoch == 0:
                    train_loader._data.patch_size_lidar = self.opt.change_patch_size_lidar
                    self.opt.patch_size_lidar = self.opt.change_patch_size_lidar
                else:
                    train_loader._data.patch_size_lidar = 1
                    self.opt.patch_size_lidar = 1

            self.train_one_epoch(train_loader)

            if self.workspace is not None:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.use_refine = False
                self.evaluate_one_epoch(valid_loader)

        self.refine(refine_loader)

        if self.use_tensorboardX:
            self.writer.close()

    def evaluate(self, loader, name=None, refine=True):
        self.use_refine = refine
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None, name=None, write_video=True, refine=True):
        if save_path is None:
            save_path = os.path.join(self.workspace, "results")

        if name is None:
            name = f"{self.name}_ep{self.epoch:04d}"

        os.makedirs(save_path, exist_ok=True)

        self.log(f"==> Start Test, save results to {save_path}")

        if write_video:
            all_preds = []
            all_preds_depth = []

        self.use_refine = refine

        pbar = tqdm.tqdm(
            total=len(loader) * loader.batch_size,
            bar_format="{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
        self.model.eval()

        with torch.no_grad():
            for i, data in enumerate(loader):
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds_raydrop, preds_intensity, preds_depth = self.test_step(data)

                pred_raydrop = preds_raydrop[0].detach().cpu().numpy()
                pred_raydrop = (np.where(pred_raydrop > 0.5, 1.0, 0.0)).reshape(
                    loader._data.H_lidar, loader._data.W_lidar
                )
                pred_raydrop = (pred_raydrop * 255).astype(np.uint8)

                pred_intensity = preds_intensity[0].detach().cpu().numpy()
                pred_intensity = (pred_intensity * 255).astype(np.uint8)

                pred_depth = preds_depth[0].detach().cpu().numpy()
                pred_lidar = pano_to_lidar(
                    pred_depth / self.opt.scale, loader._data.intrinsics_lidar
                )

                np.save(
                    os.path.join(save_path, f"test_{name}_{i+1:04d}_depth_lidar.npy"),
                    pred_lidar,
                )

                pred_depth = (pred_depth * 255).astype(np.uint8)

                if write_video:
                    all_preds.append(cv2.cvtColor(cv2.applyColorMap(pred_intensity, 1), cv2.COLOR_BGR2RGB))
                    all_preds_depth.append(cv2.cvtColor(cv2.applyColorMap(pred_depth, 20), cv2.COLOR_BGR2RGB))
                else:
                    cv2.imwrite(
                        os.path.join(save_path, f"test_{name}_{i+1:04d}_raydrop.png"),
                        pred_raydrop,
                    )
                    cv2.imwrite(
                        os.path.join(
                            save_path, f"test_{name}_{i+1:04d}_intensity.png"
                        ),
                        cv2.applyColorMap(pred_intensity, 1),
                    )
                    cv2.imwrite(
                        os.path.join(save_path, f"test_{name}_{i+1:04d}_depth.png"),
                        cv2.applyColorMap(pred_depth, 20),
                    )

                pbar.update(loader.batch_size)

        if write_video:
            all_preds = np.stack(all_preds, axis=0)
            all_preds_depth = np.stack(all_preds_depth, axis=0)
            imageio.mimwrite(
                os.path.join(save_path, f"{name}_lidar_rgb.mp4"),
                all_preds,
                fps=25,
                quality=8,
                macro_block_size=1,
            )
            imageio.mimwrite(
                os.path.join(save_path, f"{name}_depth.mp4"),
                all_preds_depth,
                fps=25,
                quality=8,
                macro_block_size=1,
            )

        self.log(f"==> Finished Test.")


    def refine(self, loader):
        if self.ema is not None:
            self.ema.copy_to() # load ema model weights
            self.ema = None    # no need for final model weights

        self.model.eval()

        raydrop_input_list = []
        raydrop_gt_list = []

        self.log("Preparing for Raydrop Refinemet ...")
        for i, data in enumerate(loader):
            rays_o_lidar = data["rays_o_lidar"]  # [B, N, 3]
            rays_d_lidar = data["rays_d_lidar"]  # [B, N, 3]
            time_lidar = data['time']
            H_lidar, W_lidar = data["H_lidar"], data["W_lidar"]
            gt_raydrop = data["images_lidar"][:, :, :, 0].unsqueeze(0)

            with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                with torch.no_grad():
                    outputs_lidar = self.model.render(
                        rays_o_lidar,
                        rays_d_lidar,
                        time_lidar,
                        staged=True,
                        max_ray_batch=4096,
                        perturb=False,
                        **vars(self.opt),
                    )

            pred_rgb_lidar = outputs_lidar["image_lidar"].reshape(-1, H_lidar, W_lidar, 2)
            pred_raydrop = pred_rgb_lidar[:, :, :, 0]
            pred_intensity = pred_rgb_lidar[:, :, :, 1]
            pred_depth = outputs_lidar["depth_lidar"].reshape(-1, H_lidar, W_lidar)

            raydrop_input = torch.cat([pred_raydrop, pred_intensity, pred_depth], dim=0).unsqueeze(0)

            raydrop_input_list.append(raydrop_input)
            raydrop_gt_list.append(gt_raydrop)
            if i % 10 == 0:
                print(f"{i+1}/{len(loader)}")

        torch.cuda.empty_cache()

        raydrop_input = torch.cat(raydrop_input_list, dim=0).cuda().float().contiguous() # [B, 3, H, W]
        raydrop_gt = torch.cat(raydrop_gt_list, dim=0).cuda().float().contiguous()       # [B, 1, H, W]

        self.model.unet.train()

        loss_total = []

        refine_bs = None # set smaller batch size (e.g. 32) if OOM and adjust epochs accordingly
        refine_epoch = 1000

        optimizer = torch.optim.Adam(self.model.unet.parameters(), lr=0.001, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, total_steps=refine_epoch)

        self.log("Start UNet Optimization ...")
        for i in range(refine_epoch):
            optimizer.zero_grad()

            if refine_bs is not None:
                idx = np.random.choice(raydrop_input.shape[0], refine_bs, replace=False)
                input = raydrop_input[idx,...]
                gt = raydrop_gt[idx,...]
            else:
                input = raydrop_input
                gt = raydrop_gt

            # random mask
            mask = torch.ones_like(input).to(input.device)
            box_num_max = 32
            box_size_y_max = int(0.1 * input.shape[2])
            box_size_x_max = int(0.1 * input.shape[3])
            for j in range(np.random.randint(box_num_max)):
                box_size_y = np.random.randint(1, box_size_y_max)
                box_size_x = np.random.randint(1, box_size_x_max)
                yi = np.random.randint(input.shape[2]-box_size_y)
                xi = np.random.randint(input.shape[3]-box_size_x)
                mask[:, :, yi:yi+box_size_y, xi:xi+box_size_x] = 0.
            input = input * mask

            raydrop_refine = self.model.unet(input)
            bce_loss = self.bce_fn(raydrop_refine, gt)
            loss = bce_loss

            loss.backward()

            loss_total.append(loss.item())

            if i % 50 == 0:
                log_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                self.log(f"[{log_time}] iter:{i}, lr:{optimizer.param_groups[0]['lr']:.6f}, raydrop loss:{loss.item()}")

            optimizer.step()
            scheduler.step()

        state = {
            "epoch": self.epoch,
            "model": self.model.state_dict()
            }
        file_path = f"{self.ckpt_path}/{self.name}_ep{self.epoch:04d}_refine.pth"
        torch.save(state, file_path)

        torch.cuda.empty_cache()

    def process_pointcloud(self, loader):
        self.log("Preparing Point Clouds ...")
        self.pc_list = {}
        self.pc_ground_list = {}
        for i, data in enumerate(loader):
            # pano to lidar
            images_lidar = data["images_lidar"]
            gt_raydrop = images_lidar[:, :, :, 0]
            gt_depth = images_lidar[:, :, :, 2] * gt_raydrop
            gt_lidar = pano_to_lidar(
                gt_depth.squeeze(0).clone().detach().cpu().numpy() / self.opt.scale, 
                loader._data.intrinsics_lidar
            )
            # remove ground
            points, ground = point_removal(gt_lidar)
            # transform
            pose = data["poses_lidar"].squeeze(0)
            pose = pose.clone().detach().cpu().numpy()
            points = points * self.opt.scale
            points = np.hstack((points, np.ones((points.shape[0], 1))))
            points = (pose @ points.T).T[:,:3]
            ground = ground * self.opt.scale
            ground = np.hstack((ground, np.ones((ground.shape[0], 1))))
            ground = (pose @ ground.T).T[:,:3]
            time_lidar = data["time"]
            frame_idx = int(time_lidar * (self.opt.num_frames - 1))
            self.pc_list[f"{frame_idx}"] = points
            self.pc_ground_list[f"{frame_idx}"] = ground
            if i % 10 == 0:
                print(f"{i+1}/{len(loader)}")

    def save_checkpoint(self, name=None, full=False, best=False, remove_old=True):
        if name is None:
            name = f"{self.name}_ep{self.epoch:04d}"

        state = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "stats": self.stats,
        }

        if full:
            state["optimizer"] = self.optimizer.state_dict()
            state["lr_scheduler"] = self.lr_scheduler.state_dict()
            state["scaler"] = self.scaler.state_dict()
            if self.ema is not None:
                state["ema"] = self.ema.state_dict()

        if not best:
            state["model"] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{name}.pth"

            torch.save(state, file_path)

            if remove_old:
                self.stats["checkpoints"].append(file_path)

                if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                    old_ckpt = self.stats["checkpoints"].pop(0)
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

        else:
            if len(self.stats["results"]) > 0:
                if (
                    self.stats["best_result"] is None
                    or self.stats["results"][-1] < self.stats["best_result"]
                ):
                    self.log(
                        f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}"
                    )
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state["model"] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()

                    torch.save(state, self.best_path)
            else:
                self.log(
                    f"[WARN] no evaluated results found, skip saving best checkpoint."
                )

    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f"{self.ckpt_path}/{self.name}_ep*.pth"))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if "model" not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(
            checkpoint_dict["model"], strict=False
        )
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        if self.ema is not None and "ema" in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict["ema"])

        if model_only:
            return

        if "stats" in checkpoint_dict:
            self.stats = checkpoint_dict["stats"]
        if "epoch" in checkpoint_dict:
            self.epoch = checkpoint_dict["epoch"]
        if "global_step" in checkpoint_dict:
            self.global_step = checkpoint_dict["global_step"]
            self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")

        if self.optimizer and "optimizer" in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict["optimizer"])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")

        if self.lr_scheduler and "lr_scheduler" in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict["lr_scheduler"])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")

        if self.scaler and "scaler" in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict["scaler"])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")
