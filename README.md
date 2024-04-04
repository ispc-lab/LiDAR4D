# LiDAR4D
<img src="https://github.com/ispc-lab/LiDAR4D/assets/51731102/7f3dd959-9b97-481e-8c13-45abbc2b712d" width=25%>

**LiDAR4D: Dynamic Neural Fields for Novel Space-time View LiDAR Synthesis**  
[Zehan Zheng](https://dyfcalid.github.io/), [Fan Lu](https://fanlu97.github.io/), Weiyi Xue, [Guang Chen](https://ispc-group.github.io/)†, Changjun Jiang  († Corresponding author)  
**CVPR 2024**

**[Paper](https://arxiv.org/abs/2404.02742) | [Project Page](https://dyfcalid.github.io/LiDAR4D)**  

This repository is the PyTorch implementation for LiDAR4D.

<img src="https://github.com/ispc-lab/LiDAR4D/assets/51731102/e23640bf-bd92-4ee0-88b4-375faf8c9b4d" width=50%>

## Changelog
2023-4-4:🔥 You can reach the preprint paper on arXiv as well as the project page.  
2023-2-27:🎉 Our paper is accepted by CVPR 2024.


## Getting started

**Code will be released in a few days.** 🚀🚀🚀

<!-- ### Installation

```bash
git clone https://github.com/ispc-lab/LiDAR4D.git
cd LiDAR4D

conda create -n lidar4d python=3.9
conda activate lidar4d

# PyTorch
# CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

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

#### KITTI-360 dataset

First, download KITTI-360 dataset from
[here](https://www.cvlibs.net/datasets/kitti-360/index.php) and put the dataset
into `data/kitti360`. Your folder structure should look like this:

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

```bash
# KITTI-360
bash run_kitti_lidar4d.sh
``` -->

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

<!-- ## Acknowledgment -->

## License
All code within this repository is under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
