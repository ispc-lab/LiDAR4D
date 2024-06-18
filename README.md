<div align="center">
<h1><img src="https://github.com/ispc-lab/LiDAR4D/assets/51731102/7f3dd959-9b97-481e-8c13-45abbc2b712d" width=25%></h1>

<h3>LiDAR4D: Dynamic Neural Fields for Novel Space-time View LiDAR Synthesis</h3>  

[Zehan Zheng](https://dyfcalid.github.io/), [Fan Lu](https://fanlu97.github.io/), Weiyi Xue, [Guang Chen](https://ispc-group.github.io/)†, Changjun Jiang  († Corresponding author)  
**CVPR 2024**


**[Paper (arXiv)](https://arxiv.org/abs/2404.02742) | [Paper (CVPR)](https://openaccess.thecvf.com/content/CVPR2024/html/Zheng_LiDAR4D_Dynamic_Neural_Fields_for_Novel_Space-time_View_LiDAR_Synthesis_CVPR_2024_paper.html) | [Project Page](https://dyfcalid.github.io/LiDAR4D) | [Video](https://www.youtube.com/watch?v=E6XyG3A3EZ8) | [Poster](https://drive.google.com/file/d/13cf0rSjCjGRyBsYOcQSa6Qf1Oe1a5QCy/view?usp=sharing) | [Slides](https://drive.google.com/file/d/1Q6yTVGoBf_nfWR4rW9RcSGlxRMufmSXc/view?usp=sharing)**  

This repository is the official PyTorch implementation for LiDAR4D.

<img src="https://github.com/ispc-lab/LiDAR4D/assets/51731102/e23640bf-bd92-4ee0-88b4-375faf8c9b4d" width=50%>
</div>

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#changelog">Changelog</a>
    </li>
    <li>
      <a href="#demo">Demo</a>
    </li>
    <li>
      <a href="#introduction">Introduction</a>
    </li>
    <li>
      <a href="#getting-started">Getting started</a>
    </li>
    <li>
      <a href="#results">Results</a>
    </li>
    <li>
      <a href="#simulation">Simulation</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
  </ol>
</details>


## Changelog
2024-6-1:🕹️ We release the simulator for easier rendering and manipulation. *Happy Children's Day and Have Fun!*   
2024-5-4:📈 We update flow fields and improve temporal interpolation.   
2024-4-13:📈 We update U-Net of LiDAR4D for better ray-drop refinement.   
2024-4-5:🚀 Code of LiDAR4D is released.  
2024-4-4:🔥 You can reach the preprint paper on arXiv as well as the project page.  
2024-2-27:🎉 Our paper is accepted by CVPR 2024.  


## Demo
<video src="https://github.com/ispc-lab/LiDAR4D/assets/51731102/34f898ec-404d-4f10-afe5-1e471df2cfe2"></video>


## Introduction
<img src="https://github.com/ispc-lab/LiDAR4D/assets/51731102/42083b63-2459-4eb9-bb8f-651eca0a1148" width=90%>  

LiDAR4D is a differentiable LiDAR-only framework for novel space-time LiDAR view synthesis, which reconstructs dynamic driving scenarios and generates realistic LiDAR point clouds end-to-end. It adopts 4D hybrid neural representations and motion priors derived from point clouds for geometry-aware and time-consistent large-scale scene reconstruction.


## Getting started


### 🛠️ Installation

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


### 📁 Dataset

#### KITTI-360 dataset ([Download](https://www.cvlibs.net/datasets/kitti-360/download.php))
We use sequence00 (`2013_05_28_drive_0000_sync`) for experiments in our paper.  

<img src="https://github.com/ispc-lab/LiDAR4D/assets/51731102/c9f5d5c5-ac48-4d54-8109-9a8b745bbca0" width=65%>  

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

### 🚀 Run LiDAR4D

Set corresponding sequence config path in `--config` and you can modify logging file path in `--workspace`. Remember to set available GPU ID in `CUDA_VISIBLE_DEVICES`.   
Run the following command:
```bash
# KITTI-360
bash run_kitti_lidar4d.sh
```


<a id="results"></a>

## 📊 Results 

**KITTI-360 *Dynamic* Dataset** (Sequences: `2350` `4950` `8120` `10200` `10750` `11400`)

<table>
<tbody align="center" valign="center">
  <tr>
    <th rowspan="2">Method</th>
    <th colspan="2">Point Cloud</th>
    <th colspan="5">Depth</th>
    <th colspan="5">Intensity</th>
  </tr>
  <tr>
    <th>CD↓</th>
    <th nowrap="true">F-Score↑</th>
    <th>RMSE↓</th>
    <th>MedAE↓</th>
    <th>LPIPS↓</th>
    <th>SSIM↑</th>
    <th>PSNR↑</th>
    <th>RMSE↓</th>
    <th>MedAE↓</th>
    <th>LPIPS↓</th>
    <th>SSIM↑</th>
    <th>PSNR↑</th>
  </tr>
  <tr>
    <td>LiDAR-NeRF</td>
    <td>0.1438</td>
    <td>0.9091</td>
    <td>4.1753</td>
    <td>0.0566</td>
    <td>0.2797</td>
    <td>0.6568</td>
    <td>25.9878</td>
    <td>0.1404</td>
    <td>0.0443</td>
    <td>0.3135</td>
    <td>0.3831</td>
    <td>17.1549</td>
  </tr>
  <tr>
    <td>LiDAR4D (Ours) †</td>
    <td><b>0.1002</b></td>
    <td><b>0.9320</b></td>
    <td><b>3.0589</b></td>
    <td><b>0.0280</b></td>
    <td><b>0.0689</b></td>
    <td><b>0.8770</b></td>
    <td><b>28.7477</b></td>
    <td><b>0.0995</b></td>
    <td><b>0.0262</b></td>
    <td><b>0.1498</b></td>
    <td><b>0.6561</b></td>
    <td><b>20.0884</b></td>
  </tr>
</tbody>
</table>

<br>

**KITTI-360 *Static* Dataset** (Sequences: `1538` `1728` `1908` `3353`)

<table>
<tbody align="center" valign="center">
  <tr>
    <th rowspan="2">Method</th>
    <th colspan="2">Point Cloud</th>
    <th colspan="5">Depth</th>
    <th colspan="5">Intensity</th>
  </tr>
  <tr>
    <th>CD↓</th>
    <th nowrap="true">F-Score↑</th>
    <th>RMSE↓</th>
    <th>MedAE↓</th>
    <th>LPIPS↓</th>
    <th>SSIM↑</th>
    <th>PSNR↑</th>
    <th>RMSE↓</th>
    <th>MedAE↓</th>
    <th>LPIPS↓</th>
    <th>SSIM↑</th>
    <th>PSNR↑</th>
  </tr>
  <tr>
    <td>LiDAR-NeRF</td>
    <td>0.0923</td>
    <td>0.9226</td>
    <td>3.6801</td>
    <td>0.0667</td>
    <td>0.3523</td>
    <td>0.6043</td>
    <td>26.7663</td>
    <td>0.1557</td>
    <td>0.0549</td>
    <td>0.4212</td>
    <td>0.2768</td>
    <td>16.1683</td>
  </tr>
  <tr>
    <td>LiDAR4D (Ours) †</td>
    <td><b>0.0834</b></td>
    <td><b>0.9312</b></td>
    <td><b>2.7413</b></td>
    <td><b>0.0367</b></td>
    <td><b>0.0995</b></td>
    <td><b>0.8484</b></td>
    <td><b>29.3359</b></td>
    <td><b>0.1116</b></td>
    <td><b>0.0335</b></td>
    <td><b>0.1799</b></td>
    <td><b>0.6120</b></td>
    <td><b>19.0619</b></td>
  </tr>
</tbody>
</table>

†: The latest results better than the paper.  
*Experiments are conducted on the NVIDIA 4090 GPU. Results may be subject to some variation and randomness.*


<a id="simulation"></a>

## 🕹️ Simulation
<img src="https://github.com/ispc-lab/LiDAR4D/assets/51731102/ada49a62-8b53-47fe-8cc0-4d99af1ebad8" width=75%>  
<!-- <img src="https://github.com/ispc-lab/LiDAR4D/assets/51731102/1b34a7b4-4238-470a-acfd-499fe697e3d1" width=75%>   -->

After reconstruction, you can use the simulator to render and manipulate LiDAR point clouds in the whole scenario. It supports dynamic scene re-play, novel LiDAR configurations (`--fov_lidar`, `--H_lidar`, `--W_lidar`) and novel trajectory (`--shift_x`, `--shift_y`, `--shift_z`).  
We also provide a simple demo setting to transform LiDAR configurations from KITTI-360 to NuScenes, using `--kitti2nus` in the bash script.    
Check the sequence config and corresponding workspace and model path (`--ckpt`).  
Run the following command:
```bash
bash run_kitti_lidar4d_sim.sh
```
The results will be saved in the workspace folder.


## Acknowledgement
We sincerely appreciate the great contribution of the following works:
- [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn/tree/master)
- [LiDAR-NeRF](https://github.com/tangtaogo/lidar-nerf)
- [NFL](https://research.nvidia.com/labs/toronto-ai/nfl/)
- [K-Planes](https://github.com/sarafridov/K-Planes)


## Citation
If you find our repo or paper helpful, feel free to support us with a star 🌟 or use the following citation:  
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
