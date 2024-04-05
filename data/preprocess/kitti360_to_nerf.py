import json
import argparse
import numpy as np
from pathlib import Path
from .kitti360_loader import KITTI360Loader


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence_id",
        type=str, 
        default="4950",
        help="choose start",
    )
    return parser


def main():
    parser = get_arg_parser()
    args = parser.parse_args()

    data_root = Path(__file__).parent.parent
    kitti_360_root = data_root / "kitti360" / "KITTI-360"
    kitti_360_parent_dir = kitti_360_root.parent

    # Specify frames and splits.
    sequence_name = "2013_05_28_drive_0000"

    sequence_id = args.sequence_id

    if sequence_id == "1538":
        s_frame_id = 1538
        e_frame_id = 1601  # Inclusive
        val_frame_ids = [1551, 1564, 1577, 1590]
    elif sequence_id == "1728":
        s_frame_id = 1728
        e_frame_id = 1791  # Inclusive
        val_frame_ids = [1741, 1754, 1767, 1780]
    elif sequence_id == "1908":
        s_frame_id = 1908
        e_frame_id = 1971  # Inclusive
        val_frame_ids = [1921, 1934, 1947, 1960]
    elif sequence_id == "3353":
        s_frame_id = 3353
        e_frame_id = 3416  # Inclusive
        val_frame_ids = [3366, 3379, 3392, 3405]

    elif sequence_id == "2350":
        s_frame_id = 2350
        e_frame_id = 2400  # Inclusive
        val_frame_ids = [2360, 2370, 2380, 2390]
    elif sequence_id == "4950":
        s_frame_id = 4950
        e_frame_id = 5000  # Inclusive
        val_frame_ids = [4960, 4970, 4980, 4990]
    elif sequence_id == "8120":
        s_frame_id = 8120
        e_frame_id = 8170  # Inclusive
        val_frame_ids = [8130, 8140, 8150, 8160]
    elif sequence_id == "10200":
        s_frame_id = 10200
        e_frame_id = 10250  # Inclusive
        val_frame_ids = [10210, 10220, 10230, 10240]
    elif sequence_id == "10750":
        s_frame_id = 10750
        e_frame_id = 10800  # Inclusive
        val_frame_ids = [10760, 10770, 10780, 10790]
    elif sequence_id == "11400":
        s_frame_id = 11400
        e_frame_id = 11450  # Inclusive
        val_frame_ids = [11410, 11420, 11430, 11440]
    else:
        raise ValueError(f"Invalid sequence id: {sequence_id}")

    print(f"Using sequence {s_frame_id}-{e_frame_id}")

    frame_ids = list(range(s_frame_id, e_frame_id + 1))
    num_frames = len(frame_ids)

    test_frame_ids = val_frame_ids
    train_frame_ids = [x for x in frame_ids if x not in val_frame_ids]

    # Load KITTI-360 dataset.
    k3 = KITTI360Loader(kitti_360_root)

    # Get lidar paths (range view not raw data).
    range_view_dir = kitti_360_parent_dir / "train"
    range_view_paths = [
        range_view_dir / "{:010d}.npy".format(int(frame_id)) for frame_id in frame_ids
    ]

    # Get lidar2world.
    lidar2world = k3.load_lidars(sequence_name, frame_ids)

    # Get image dimensions, assume all images have the same dimensions.
    lidar_range_image = np.load(range_view_paths[0])
    lidar_h, lidar_w, _ = lidar_range_image.shape

    # Split by train/test/val.
    all_indices = [i - s_frame_id for i in frame_ids]
    train_indices = [i - s_frame_id for i in train_frame_ids]
    val_indices = [i - s_frame_id for i in val_frame_ids]
    test_indices = [i - s_frame_id for i in test_frame_ids]

    split_to_all_indices = {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }
    for split, indices in split_to_all_indices.items():
        print(f"Split {split} has {len(indices)} frames.")
        id_split = [frame_ids[i] for i in indices]
        lidar_paths_split = [range_view_paths[i] for i in indices]
        lidar2world_split = [lidar2world[i] for i in indices]

        json_dict = {
            "w_lidar": lidar_w,
            "h_lidar": lidar_h,
            "num_frames": num_frames,
            "num_frames_split": len(id_split),
            "frames": [
                {
                    "frame_id": id,
                    "lidar_file_path": str(
                        lidar_path.relative_to(kitti_360_parent_dir)
                    ),
                    "lidar2world": lidar2world.tolist(),
                }
                for (
                    id,
                    lidar_path,
                    lidar2world,
                ) in zip(
                    id_split,
                    lidar_paths_split,
                    lidar2world_split,
                )
            ],
        }
        json_path = kitti_360_parent_dir / f"transforms_{sequence_id}_{split}.json"

        with open(json_path, "w") as f:
            json.dump(json_dict, f, indent=2)
            print(f"Saved {json_path}.")


if __name__ == "__main__":
    main()
