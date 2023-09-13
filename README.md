# OctFormer: Octree-based Transformers for 3D Point Clouds

This repository contains the implementation of **OctFormer**. The code is
released under the **MIT license**. The code has been awarded the [Replicability Stamp](http://www.replicabilitystamp.org#https-github-com-octree-nn-octformer) by the [Graphics Replicability Stamp Initiative](https://www.replicabilitystamp.org/).


**[OctFormer: Octree-based Transformers for 3D Point Clouds](https://wang-ps.github.io/octformer.html)**<br/>
[Peng-Shuai Wang](https://wang-ps.github.io/)<br/>
ACM Transactions on Graphics (SIGGRAPH), 42(4), 2023

![teaser](teaser.png)


- [OctFormer: Octree-based Transformers for 3D Point Clouds](#octformer-octree-based-transformers-for-3d-point-clouds)
  - [1. Installation](#1-installation)
  - [2. ScanNet Segmentation](#2-scannet-segmentation)
  - [3. ScanNet200 Segmentation](#3-scannet200-segmentation)
  - [4. SUN RGB-D Detection](#4-sun-rgb-d-detection)
  - [5. ModelNet40 Classification](#5-modelnet40-classification)
  - [6. Citation](#6-citation)


## 1. Installation

The code has been tested on Ubuntu 20.04 with 4 Nvidia 3090 GPUs (24GB memory).


1. Install [Conda](https://www.anaconda.com/) and create a `Conda` environment.

    ```bash
    conda create --name octformer python=3.8
    conda activate octformer
    ```

2. Install PyTorch-1.12.1 with conda according to the official documentation.

    ```bash
    conda install pytorch==1.12.1 torchvision torchaudio cudatoolkit=11.3 -c pytorch
    ```

3. Clone this repository and install the requirements.

    ```bash
    git clone https://github.com/octree-nn/octformer.git
    cd  octformer
    pip install -r requirements.txt
    ```

4. Install the library for octree-based depthwise convolution.

    ```bash
    git clone https://github.com/octree-nn/dwconv.git
    pip install ./dwconv
    ```

5. To run the detection experiments,
   [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) is required.
   And the code is tested with `mmdet3d==1.0.0rc5`. Run the following commands
   to install it. For detailed instructions, please refer to the
   [official documentation](https://mmdetection3d.readthedocs.io/en/latest/get_started.html#installation).
   Make sure the results of
   [FCAF3D](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/fcaf3d/README.md)
   can be reproduced before running the experiments.

    ```bash
    pip install openmim==0.3.3
    mim install mmcv-full==1.6.2
    mim install mmdet==2.26.0
    mim install mmsegmentation==0.29.1
    git clone https://github.com/open-mmlab/mmdetection3d.git
    cd mmdetection3d
    git checkout v1.0.0rc5
    pip install -e .
    ```

## 2. ScanNet Segmentation

1. **Data**: Download the data from the
   [ScanNet benchmark](https://kaldir.vc.in.tum.de/scannet_benchmark/).
   Unzip the data and place it to the folder <scannet_folder>. Run the following
   command to prepare the dataset.

    ```bash
    python tools/seg_scannet.py --run process_scannet --path_in <scannet_folder>
    ```

2. **Train**: Run the following command to train the network with 4 GPUs and
   port 10001. The mIoU on the validation set without voting is 74.8. The
   training takes less than 16h on 4 Nvidia 3090 GPUs. And the training log and
   weights can be downloaded
   [here](https://1drv.ms/u/s!Ago-xIr0OR2-gRrV35QGxnHJR4ku?e=ZXRqV7).

    ```bash
    python scripts/run_seg_scannet.py --gpu 0,1,2,3 --alias scannet --port 10001
    ```

3. **Evaluate**: Run the following command to get the per-point predictions for
   the validation dataset with a voting strategy. And after voting, the mIoU is
   76.3 on the validation dataset.

    ```bash
    python scripts/run_seg_scannet.py --gpu 0 --alias scannet --run validate
    ```


## 3. ScanNet200 Segmentation


1. **Data**: Download the data from the
   [ScanNet benchmark](https://kaldir.vc.in.tum.de/scannet_benchmark/).
   Unzip the data and place it to the folder <scannet_folder>. Run the following
   command to prepare the dataset.

    ```bash
    python tools/seg_scannet200.py --run process_scannet --path_in <scannet_folder>  \
           --path_out data/scanet200.npz  --align_axis  --scannet200
    ```

2. **Train**: Run the following command to train the network with 4 GPUs. The
    mIoU on the validation set without voting is 31.7, the training log and
   weights can be downloaded
   [here](https://1drv.ms/u/s!Ago-xIr0OR2-gRwsNivzRalw0M4S?e=b92sv6).
   With OctFormer-Large, the mIoU increases to 32.2.

    ```bash
    python scripts/run_seg_scannet200.py --gpu 0,1,2,3 --alias scannet200
    ```

3. **Evaluate**: Run the following command to get the per-point predictions for
   the validation dataset with a voting strategy. And after voting, the mIoU is
   32.6 on the validation dataset.

    ```bash
    python scripts/run_seg_scannet200.py --gpu 0 --alias scannet200 --run validate
    ```


## 4. SUN RGB-D Detection

1. **Data**: Prepare the data according to the
   [official documentation](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/sunrgbd.html)
   of mmdetection3d. Denote the path to the data as <sunrgbd_folder>. Run the
   following command to build a symbolic link to the data.

    ```bash
    ln -s <sunrgbd_folder> data/sunrgbd
    ```

2. **Training**: Run the following command to train the network with 4 GPUs. The
    maximum mAP@0.25 and mAP@05 on the validation set are 66.6 and 50.6,
    respectively. The training log and weights can be downloaded
    [here](https://1drv.ms/u/s!Ago-xIr0OR2-gR70qXLpohtYonuP?e=wlWMqm).

    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  \
        --master_port=29000  detection.py  configs/det_sunrgbd.py   --launcher=pytorch  \
        --work-dir=logs/sunrgbd/octformer
    ```

## 5. ModelNet40 Classification

1. **Data**: Run the following command to prepare the dataset.

    ```bash
    python tools/cls_modelnet.py
    ```

2. **Train**: Run the following command to train the network with 1 GPU. The
   classification accuracy on the testing set without voting is 92.7%. And the
   training log and weights can be downloaded
   [here](https://1drv.ms/u/s!Ago-xIr0OR2-gRskk20x7V_Mc9FI?e=jVAf8o).
    ```bash
    python classification.py --config configs/cls_m40.yaml SOLVER.gpu 0,
    ```

## 6. Citation

   ```bibtex 
    @article {Wang2023OctFormer,
        title      = {OctFormer: Octree-based Transformers for {3D} Point Clouds},
        author     = {Wang, Peng-Shuai},
        journal    = {ACM Transactions on Graphics (SIGGRAPH)},
        volume     = {42},
        number     = {4},
        year       = {2023},
    }
   ```
