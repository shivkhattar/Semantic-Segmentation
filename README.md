# Semantic Segmentation using Deep Convolutional Networks

## Introduction

In this repo, we try to perform semantic segmentation using various deep convolutional networks like U-Net, FPN, PSPNET and Deeplab. This project was done as a part of the final project for the course Computer Vision at NYU Courant in Fall 2020.


## Project Benchmarks

All the models were trained on the PASCAL VOC 2012 dataset and evaluated on the PASCAL VOC 2012 
Note: All models are trained only on PASCAL VOC 2012 trainaug dataset and evaluated on PASCAL VOC 2012 val dataset.

| Architecture | Accuracy | Mean IOU |
|:---:|:---:|:---:|
| DeepLab | 94.58% | 76.47% |
| PSPNet | 94.09% | 74.01% |
| FPN  | 93.95% | 73.54% |
| U-Net | 93.35% | 70.77% |


## Installation
### Requirements

- Linux
- Python 3.6+
- PyTorch 1.4.0 or higher
- CUDA 10.0


### Install

1. Create a conda virtual environment and activate it.

```shell
conda create -n semanticsegmentation python=3.6.9 -y
conda activate semanticsegmentation
```

2. Install dependencies using the command:

```shell
pip install -r requirements.txt
```

3. Install PyTorch and torchvision using the command:

```shell
conda install pytorch==1.4.0 cudatoolkit=10.0 torchvision -c pytorch
```

## Train

Train the data using the following commands:

1. For U-Net:
```shell
python train_unet.py
```
2. For FPN:
```shell
python train_fpn.py
```
3. For PSPNet:
```shell
python train_pspnet.py
```
4. For Deeplab:
```shell
python train_deeplab.py
```

Parameters can be fine-tuned in these files as well. Provide the link to the data in the train python files, or make sure data is present in `data/`.
Model snapshots and logs will be saved at `out/`. 


## Inference

Results can be visualized by running the models for a selected image by providing the path to the model. Use the following command to run inference.

1. For U-Net:
```shell
python tools/infer_unet.py model_checkpoint_path image_path --out folder_name
```
2. For FPN:
```shell
python tools/infer_fpn.py model_checkpoint_path image_path --out folder_name
```
3. For PSPNet:
```shell
python tools/infer_pspnet.py model_checkpoint_path image_path --out folder_name
```
4. For Deeplab:
```shell
python tools/infer_deeplab.py model_checkpoint_path image_path --out folder_name
```

Results get saved in the folder `save_result`. Optionally a custom path can be provided by using the argument `--out`.

## Test

Test the model using the following commands. Please note that the tests run on the validation dataset from Pascal VOC 2012.

1. For U-Net:
```shell
python tools/test_unet.py model_checkpoint_path
```
2. For FPN:
```shell
python tools/test_fpn.py model_checkpoint_path
```
3. For PSPNet:
```shell
python tools/test_pspnet.py model_checkpoint_path
```
4. For Deeplab:
```shell
python tools/test_deeplab.py model_checkpoint_path
```
