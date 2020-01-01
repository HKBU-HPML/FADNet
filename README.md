# FADNet: A Fast and Accurate Network for Disparity Estimation

Updating...

This repository contains the code (in PyTorch) for "[FADNet]()" paper.

## Contents

1. [Introduction](#introduction)
2. [Usage](#usage)
3. [Results](#results)
4. [Acknowledgement](#acknowledgement)
5. [Contacts](#contacts)


## Introduction
We propose an efficient and accurate deep network for disparity estimation named FADNet with three main features: 
- It exploits efficient 2D based correlation layers with stacked blocks to preserve fast computation.
- It combines the residual structures to make the deeper model easier to learn.
- It contains multi-scale predictions so as to exploit a multi-scale weight scheduling training technique to improve the accuracy.

## Usage

### Dependencies

- [Python2.7](https://www.python.org/downloads/)
- [PyTorch(1.2.0+)](http://pytorch.org)
- torchvision 0.2.0 (higher version may cause issues)
- [KITTI Stereo](http://www.cvlibs.net/datasets/kitti/eval_stereo.php)
- [Scene Flow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

```
Usage of Scene Flow dataset
Download RGB cleanpass images and its disparity for three subset: FlyingThings3D, Driving, and Monkaa.
Put them in the same folder. Retain the file names as well as the file organization.
```

### Train
We use template scripts to configure the training task, which are stored in exp_configs. One sample "fadnet.conf" is as follows:
```
net=fadnet
loss=loss_configs/fadnet_sceneflow.json
outf_model=models/${net}-sceneflow
logf=logs/${net}-sceneflow.log

lr=1e-4
devices=0,1,2,3
dataset=sceneflow
trainlist=lists/SceneFlow.list
vallist=lists/FlyingThings3D_release_TEST.list
startR=0
startE=0
batchSize=16
maxdisp=-1
model=none
#model=fadnet_sceneflow.pth
```
| Parameter | Description | Options |
|---|---|---|
| net | network architecture name | dispnets, dispnetc, dispnetcss, fadnet, psmnet, ganet |
| loss | loss weight scheduling configuration file | depends on the training scheme |
| outf_model | folder name to store the model files | \ |
| logf | log file name | \ |
| lr | initial learning rate | \ |
| devices | GPU device IDs to use | depends on the hardware system |
| dataset | dataset name to train | sceneflow |
| train(val)list | sample lists for training/validation | \ |
| startR | the round index to start training (for restarting training from the checkpoint) | \ |
| startE | the epoch index to start training (for restarting training from the checkpoint) | \ |
| batchSize | the number of samples per batch | \ |
| maxdisp | the maximum disparity that the model tries to predict | \ |
| model | the model file path of the checkpoint | \ |


### Evaluation

### Pretrained Model

## Results

## Acknowledgement
We acknowledge the following repositories and papers since our project has used some codes of them. 
- [PSMNet](https://github.com/JiaRenChang/PSMNet) from [Jia-Ren Chang](https://github.com/JiaRenChang)
- [GANet](https://github.com/feihuzhang/GANet) from [Feihu Zhang](https://github.com/feihuzhang)
- [PWCNet](https://github.com/NVlabs/PWC-Net/tree/master/PyTorch) from [NVIDIA Research Projects](https://github.com/NVlabs)

```
@inproceedings{chang2018pyramid,
  title={Pyramid Stereo Matching Network},
  author={Chang, Jia-Ren and Chen, Yong-Sheng},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={5410--5418},
  year={2018}
}
```
```
@inproceedings{Zhang2019GANet,
  title={GA-Net: Guided Aggregation Net for End-to-end Stereo Matching},
  author={Zhang, Feihu and Prisacariu, Victor and Yang, Ruigang and Torr, Philip HS},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={185--194},
  year={2019}
}
```
```
@InProceedings{Sun2018PWC-Net,
  author    = {Deqing Sun and Xiaodong Yang and Ming-Yu Liu and Jan Kautz},
  title     = {{PWC-Net}: {CNNs} for Optical Flow Using Pyramid, Warping, and Cost Volume},
  booktitle = CVPR,
  year      = {2018},
}
```
## Contacts
qiangwang@comp.hkbu.edu.hk

Any discussions or concerns are welcomed!
