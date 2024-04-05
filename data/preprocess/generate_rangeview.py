import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from utils.convert import lidar_to_pano_with_intensities


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


def LiDAR_2_Pano_KITTI(
    local_points_with_intensities, lidar_H, lidar_W, intrinsics, max_depth=80.0
):
    pano, intensities = lidar_to_pano_with_intensities(
        local_points_with_intensities=local_points_with_intensities,
        lidar_H=lidar_H,
        lidar_W=lidar_W,
        lidar_K=intrinsics,
        max_depth=max_depth,
    )
    range_view = np.zeros((lidar_H, lidar_W, 3))
    range_view[:, :, 1] = intensities
    range_view[:, :, 2] = pano
    return range_view


def generate_train_data(
    H,
    W,
    intrinsics,
    lidar_paths,
    out_dir,
    points_dim,
):
    """
    Args:
        H: Heights of the range view.
        W: Width of the range view.
        intrinsics: (fov_up, fov) of the range view.
        out_dir: Output directory.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for lidar_path in tqdm(lidar_paths):
        point_cloud = np.fromfile(lidar_path, dtype=np.float32)
        point_cloud = point_cloud.reshape((-1, points_dim))
        pano = LiDAR_2_Pano_KITTI(point_cloud, H, W, intrinsics)
        frame_name = lidar_path.split("/")[-1]
        suffix = frame_name.split(".")[-1]
        frame_name = frame_name.replace(suffix, "npy")
        np.save(out_dir / frame_name, pano)


def create_kitti_rangeview(frame_start, frame_end):
    data_root = Path(__file__).parent.parent
    kitti_360_root = data_root / "kitti360" / "KITTI-360"
    kitti_360_parent_dir = kitti_360_root.parent
    out_dir = kitti_360_parent_dir / "train"
    sequence_name = "2013_05_28_drive_0000"

    H = 66
    W = 1030
    intrinsics = (2.0, 26.9)  # fov_up, fov

    frame_ids = list(range(frame_start, frame_end + 1))

    lidar_dir = (
        kitti_360_root
        / "data_3d_raw"
        / f"{sequence_name}_sync"
        / "velodyne_points"
        / "data"
    )
    lidar_paths = [
        os.path.join(lidar_dir, "%010d.bin" % frame_id) for frame_id in frame_ids
    ]

    generate_train_data(
        H=H,
        W=W,
        intrinsics=intrinsics,
        lidar_paths=lidar_paths,
        out_dir=out_dir,
        points_dim=4,
    )


def main():
    parser = get_arg_parser()
    args = parser.parse_args()

    # Check dataset.
    if args.dataset == "kitti360":
        frame_start = args.sequence_id
        if args.sequence_id == "1538":
            frame_start = 1538
            frame_end = 1601
        elif args.sequence_id == "1728":
            frame_start = 1728
            frame_end = 1791
        elif args.sequence_id == "1908":
            frame_start = 1908
            frame_end = 1971
        elif args.sequence_id == "3353":
            frame_start = 3353
            frame_end = 3416
        
        elif args.sequence_id == "2350":
            frame_start = 2350
            frame_end = 2400
        elif args.sequence_id == "4950":
            frame_start = 4950
            frame_end = 5000
        elif args.sequence_id == "8120":
            frame_start = 8120
            frame_end = 8170
        elif args.sequence_id == "10200":
            frame_start = 10200
            frame_end = 10250
        elif args.sequence_id == "10750":
            frame_start = 10750
            frame_end = 10800
        elif args.sequence_id == "11400":
            frame_start = 11400
            frame_end = 11450
        else:
            raise ValueError(f"Invalid sequence id: {sequence_id}")
        
        print(f"Generate rangeview from {frame_start} to {frame_end} ...")
        create_kitti_rangeview(frame_start, frame_end)


if __name__ == "__main__":
    main()
