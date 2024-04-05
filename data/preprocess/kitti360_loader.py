from pathlib import Path
import numpy as np
import camtools as ct


class KITTI360Loader:
    def __init__(self, kitti_360_root) -> None:
        # Root directory.
        self.kitti_360_root = Path(kitti_360_root)
        if not self.kitti_360_root.is_dir():
            raise FileNotFoundError(f"KITTI-360 {kitti_360_root} not found.")

        # Other directories.
        self.calibration_dir = self.kitti_360_root / "calibration"
        self.data_poses_dir = self.kitti_360_root / "data_poses"
        self.data_3d_raw_dir = self.kitti_360_root / "data_3d_raw"

        # Check if all directories exist.
        if not self.calibration_dir.is_dir():
            raise FileNotFoundError(
                f"Calibration dir {self.calibration_dir} not found."
            )
        if not self.data_poses_dir.is_dir():
            raise FileNotFoundError(
                f"Data poses dir {self.data_poses_dir} not found."
            )
        if not self.data_3d_raw_dir.is_dir():
            raise FileNotFoundError(
                f"Data 3D raw dir {self.data_3d_raw_dir} not found."
            )

    @staticmethod
    def _read_variable(fid, name, M, N):
        """
        Ref:
            kitti360scripts/devkits/commons/loadCalibration.py
        """
        # Rewind
        fid.seek(0, 0)

        # Search for variable identifier
        line = 1
        success = 0
        while line:
            line = fid.readline()
            if line.startswith(name):
                success = 1
                break

        # Return if variable identifier not found
        if success == 0:
            return None

        # Fill matrix
        line = line.replace("%s:" % name, "")
        line = line.split()
        assert len(line) == M * N
        line = [float(x) for x in line]
        mat = np.array(line).reshape(M, N)

        return mat

    def _load_all_lidars(self, sequence_name):
        """
        Args:
            sequence_name: str, name of sequence. e.g. "2013_05_28_drive_0000".

        Returns:
            velo_to_world: 4x4 metric.
        """
        data_poses_dir = self.data_poses_dir / f"{sequence_name}_sync"
        assert data_poses_dir.is_dir()

        # IMU to world transformation (poses.txt).
        poses_path = data_poses_dir / "poses.txt"
        imu_to_world_dict = dict()
        frame_ids = []
        for line in np.loadtxt(poses_path):
            frame_id = int(line[0])
            frame_ids.append(frame_id)
            imu_to_world = line[1:].reshape((3, 4))
            imu_to_world_dict[frame_id] = imu_to_world

        # Camera to IMU transformation (calib_cam_to_pose.txt).
        cam_to_imu_path = self.calibration_dir / "calib_cam_to_pose.txt"
        with open(cam_to_imu_path, "r") as fid:
            cam_00_to_imu = KITTI360Loader._read_variable(fid, "image_00", 3, 4)
            cam_00_to_imu = ct.convert.pad_0001(cam_00_to_imu)

        # Camera00 to Velo transformation (calib_cam_to_velo.txt).
        cam00_to_velo_path = self.calibration_dir / "calib_cam_to_velo.txt"
        with open(cam00_to_velo_path, "r") as fid:
            line = fid.readline().split()
            line = [float(x) for x in line]
            cam_00_to_velo = np.array(line).reshape(3, 4)
            cam_00_to_velo = ct.convert.pad_0001(cam_00_to_velo)

        # Compute velo_to_world
        velo_to_world_dict = dict()
        for frame_id in frame_ids:
            imu_to_world = imu_to_world_dict[frame_id]
            cam_00_to_world_unrec = imu_to_world @ cam_00_to_imu
            velo_to_world = cam_00_to_world_unrec @ np.linalg.inv(cam_00_to_velo)
            velo_to_world_dict[frame_id] = ct.convert.pad_0001(velo_to_world)

        return velo_to_world_dict

    def load_lidars(self, sequence_name, frame_ids):
        """
        Args:
            sequence_name: str, name of sequence. e.g. "2013_05_28_drive_0000".
            frame_ids: list of int, frame ids. e.g. range(1908, 1971+1).

        Returns:
            velo_to_worlds
        """
        velo_to_world_dict = self._load_all_lidars(sequence_name)
        # velo_to_worlds = [velo_to_world_dict[frame_id] for frame_id in frame_ids]
        velo_to_worlds = []
        for frame_id in frame_ids:
            if frame_id in velo_to_world_dict.keys():
                velo_to_worlds.append(velo_to_world_dict[frame_id])
                tmp = velo_to_world_dict[frame_id]
            else:
                velo_to_worlds.append(tmp)
        velo_to_worlds = np.stack(velo_to_worlds)
        return velo_to_worlds
