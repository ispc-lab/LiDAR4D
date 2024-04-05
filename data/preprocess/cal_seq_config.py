import argparse
import numpy as np
np.set_printoptions(suppress=True)
import os
import json
import tqdm
from utils.convert import pano_to_lidar


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="kitti360",
        choices=["kitti360"],
        help="The dataset loader to use.",
    )
    parser.add_argument(
        "--sequence_id",
        type=str, 
        default="4950",
        help="choose start",
    )
    return parser


def cal_centerpose_bound_scale(
    lidar_rangeview_paths, lidar2worlds, fov_lidar, bound=1.0
):
    near = 200
    far = 0
    points_world_list = []
    for i, lidar_rangeview_path in enumerate(lidar_rangeview_paths):
        pano = np.load(lidar_rangeview_path)
        point_cloud = pano_to_lidar(pano=pano[:, :, 2], lidar_K=fov_lidar)
        point_cloud = np.concatenate(
            [point_cloud, np.ones(point_cloud.shape[0]).reshape(-1, 1)], -1
        )
        dis = np.sqrt(
            point_cloud[:, 0] ** 2 + point_cloud[:, 1] ** 2 + point_cloud[:, 2] ** 2
        )
        near = min(min(dis), near)
        far = max(far, max(dis))
        points_world = (point_cloud @ lidar2worlds[i].T)[:, :3]
        points_world_list.append(points_world)
    print("near, far:", near, far)

    pc_all_w = np.concatenate(points_world_list)[:, :3]

    centerpose = [
        (np.max(pc_all_w[:, 0]) + np.min(pc_all_w[:, 0])) / 2.0,
        (np.max(pc_all_w[:, 1]) + np.min(pc_all_w[:, 1])) / 2.0,
        (np.max(pc_all_w[:, 2]) + np.min(pc_all_w[:, 2])) / 2.0,
    ]
    print("centerpose: ", centerpose)
    pc_all_w_centered = pc_all_w - centerpose

    bound_ori = [
        np.max(pc_all_w_centered[:, 0]),
        np.max(pc_all_w_centered[:, 1]),
        np.max(pc_all_w_centered[:, 2]),
    ]
    scale = bound / np.max(bound_ori)
    print("scale: ", scale)
    
    return scale, centerpose


def get_path_pose_from_json(root_path, sequence_id):
    with open(
        os.path.join(root_path, f"transforms_{sequence_id}_train.json"), "r"
    ) as f:
        transform = json.load(f)
    num_frames = transform["num_frames"]
    frames = transform["frames"]
    poses_lidar = []
    paths_lidar = []
    for f in tqdm.tqdm(frames, desc=f"Loading {type} data"):
        pose_lidar = np.array(f["lidar2world"], dtype=np.float32)  # [4, 4]
        f_lidar_path = os.path.join(root_path, f["lidar_file_path"])
        poses_lidar.append(pose_lidar)
        paths_lidar.append(f_lidar_path)
    return paths_lidar, poses_lidar, num_frames


def main():
    parser = get_arg_parser()
    args = parser.parse_args()

    # kitti360
    root_path = f"data/{args.dataset}"

    sequence_id = args.sequence_id

    lidar_rangeview_paths, lidar2worlds, num_frames = get_path_pose_from_json(
        root_path, sequence_id=sequence_id
    )
    fov_lidar = [2.0, 26.9]  # fov_up, fov

    scale, centerpose = cal_centerpose_bound_scale(lidar_rangeview_paths, lidar2worlds, fov_lidar)

    config_path = f"configs/{args.dataset}_{args.sequence_id}.txt"
    with open(config_path, "w") as f:
        f.write("dataloader = {}\n".format(args.dataset))
        f.write("path = {}\n".format(root_path))
        f.write("sequence_id = {}\n".format(sequence_id))
        f.write("num_frames = {}\n".format(num_frames))
        f.write("fov_lidar = {}\n".format(fov_lidar))
        f.write("scale = {}\n".format(scale))
        f.write("offset = {}\n".format(centerpose))


if __name__ == "__main__":
    main()
