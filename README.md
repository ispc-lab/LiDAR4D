# LiDAR4D
<img src="https://github.com/ispc-lab/LiDAR4D/assets/51731102/7f3dd959-9b97-481e-8c13-45abbc2b712d" width=25%>

**LiDAR4D: Dynamic Neural Fields for Novel Space-time View LiDAR Synthesis**  
[Zehan Zheng](https://dyfcalid.github.io/), [Fan Lu](https://fanlu97.github.io/), Weiyi Xue, [Guang Chen](https://ispc-group.github.io/)†, Changjun Jiang  († Corresponding author)  
**CVPR 2024**

**[Paper](https://arxiv.org/abs/2404.02742) | [Project Page](https://dyfcalid.github.io/LiDAR4D)**  

This repository is the official PyTorch implementation for LiDAR4D.

<img src="https://github.com/ispc-lab/LiDAR4D/assets/51731102/e23640bf-bd92-4ee0-88b4-375faf8c9b4d" width=50%>

## Changelog
2023-4-13:📈 We update U-Net of LiDAR4D for better ray-drop refinement.   
2023-4-5:🚀 Code of LiDAR4D is released.  
2023-4-4:🔥 You can reach the preprint paper on arXiv as well as the project page.  
2023-2-27:🎉 Our paper is accepted by CVPR 2024.  


## Introduction
<img src="https://github.com/ispc-lab/LiDAR4D/assets/51731102/42083b63-2459-4eb9-bb8f-651eca0a1148" width=90%>  

LiDAR4D is a differentiable LiDAR-only framework for novel space-time LiDAR view synthesis, which reconstructs dynamic driving scenarios and generates realistic LiDAR point clouds end-to-end. It adopts 4D hybrid neural representations and motion priors derived from point clouds for geometry-aware and time-consistent large-scale scene reconstruction.


## Getting started


### Installation

```bash
git clone https://github.com/ispc-lab/LiDAR4D.git
cd LiDAR4D

conda create -n lidar4d python=3.9
conda activate lidar4d

# PyTorch
# CUDA 12.1
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8
# pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
# CUDA <= 11.7
# pip install torch==2.0.0 torchvision torchaudio

# Dependencies
pip install -r requirements.txt

# Local compile for tiny-cuda-nn
git clone --recursive https://github.com/nvlabs/tiny-cuda-nn
cd tiny-cuda-nn/bindings/torch
python setup.py install

# compile packages in utils
cd utils/chamfer3D
python setup.py install
```


### Dataset

#### KITTI-360 dataset ([Download](https://www.cvlibs.net/datasets/kitti-360/download.php))
We use sequence00 (`2013_05_28_drive_0000_sync`) for experiments in our paper.  

<img src="https://github.com/ispc-lab/LiDAR4D/assets/51731102/c9f5d5c5-ac48-4d54-8109-9a8b745bbca0" width=50%>  

Download KITTI-360 dataset (2D images are not needed) and put them into `data/kitti360`.  
(or use symlinks: `ln -s DATA_ROOT/KITTI-360 ./data/kitti360/`).  
The folder tree is as follows:  

```bash
data
└── kitti360
    └── KITTI-360
        ├── calibration
        ├── data_3d_raw
        └── data_poses
```

Next, run KITTI-360 dataset preprocessing: (set `DATASET` and `SEQ_ID`)  

```bash
bash preprocess_data.sh
```

After preprocessing, your folder structure should look like this:  

```bash
configs
├── kitti360_{sequence_id}.txt
data
└── kitti360
    ├── KITTI-360
    │   ├── calibration
    │   ├── data_3d_raw
    │   └── data_poses
    ├── train
    ├── transforms_{sequence_id}test.json
    ├── transforms_{sequence_id}train.json
    └── transforms_{sequence_id}val.json
```

### Run LiDAR4D

Set corresponding sequence config path in `--config` and you can modify logging file path in `--workspace`. Remember to set available GPU ID in `CUDA_VISIBLE_DEVICES`.   
Run the following command:
```bash
# KITTI-360
bash run_kitti_lidar4d.sh
```

## Acknowledgment
We sincerely appreciate the great contribution of the following works:
- [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn/tree/master)
- [LiDAR-NeRF](https://github.com/tangtaogo/lidar-nerf)
- [NFL](https://research.nvidia.com/labs/toronto-ai/nfl/)
- [K-Planes](https://github.com/sarafridov/K-Planes)


## Citation
Please use the following citation if you find our repo or paper helps:  
```bibtex
@inproceedings{zheng2024lidar4d,
  title     = {LiDAR4D: Dynamic Neural Fields for Novel Space-time View LiDAR Synthesis},
  author    = {Zheng, Zehan and Lu, Fan and Xue, Weiyi and Chen, Guang and Jiang, Changjun},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2024}
  }
```


## License
All code within this repository is under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
