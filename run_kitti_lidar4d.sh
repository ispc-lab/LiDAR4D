#! /bin/bash
CUDA_VISIBLE_DEVICES=0 python main_lidar4d.py \
--config configs/kitti360_4950.txt \
--workspace log/kitti360_lidar4d_f4950_release \
--lr 1e-2 \
--num_rays_lidar 1024 \
--iters 30000 \
--alpha_d 1 \
--alpha_i 0.1 \
--alpha_r 0.01 \

# --refine
# --test_eval
