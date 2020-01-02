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
Download RGB cleanpass images and its disparity for three subset: FlyingThings3D, Driving, and Monkaa. Organize them as follows:
- FlyingThings3D_release/frames_cleanpass
- FlyingThings3D_release/disparity
- driving_release/frames_cleanpass
- driving_release/disparity
- monkaa_release/frames_cleanpass
- monkaa_release/disparity
Put them in the data/ folder (or soft link). The *train.sh* defaultly locates the data root path as data/.
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

We have integrated PSMNet and GANet for comparison. The sample configuration files are also given. 

To start training, use the following command, *dnn=CONFIG_FILE sh train.sh*, such as:
```
dnn=fadnet sh train.sh
```
You do not need the suffix for *CONFIG_FILE*. 

### Evaluation
We have two modes for performance evaluation, *test* and *detect*, respectively. *test* requires that the testing samples should have ground truth of disparity and then reports the average End-point-error (EPE). *detect* does not require any ground truth for EPE computation. However, *detect* stores the disparity maps for each sample in the given list. 

For the *test* mode, one can revise *test.sh* and run *sh test.sh*. The contents of *test.sh* are as follows:
```
net=fadnet
maxdisp=-1
dataset=sceneflow
trainlist=lists/SceneFlow.list
vallist=lists/FlyingThings3D_release_TEST.list

loss=loss_configs/test.json
outf_model=models/test/
logf=logs/${net}_test_on_${dataset}.log

lr=1e-4
devices=0,1,2,3
startR=0
startE=0
batchSize=8
model=models/fadnet.pth
python main.py --cuda --net $net --loss $loss --lr $lr \
               --outf $outf_model --logFile $logf \
               --devices $devices --batch_size $batchSize \
               --trainlist $trainlist --vallist $vallist \
               --dataset $dataset --maxdisp $maxdisp \
               --startRound $startR --startEpoch $startE \
               --model $model 
```
Most of the parameters in *test.sh* are similar to training. However, you can just ignore parameters, including *trainlist*, *loss*, *outf_model*, since they are not used in the *test* mode.

For the *detect* mode, one can revise *detect.sh* and run *sh detect.sh*. The contents of *detect.sh* are as follows:
```
net=fadnet
dataset=sceneflow

model=models/fadnet.pth
outf=detect_results/${net}-${dataset}/

filelist=lists/FlyingThings3D_release_TEST.list
filepath=data

CUDA_VISIBLE_DEVICES=0 python detecter.py --model $model --rp $outf --filelist $filelist --filepath $filepath --devices 0 --net ${net} 
```
You can revise the value of *outf* to change the folder that stores the predicted disparity maps.

### Finetuning on KITTI datasets and result submission
We re-use the codes in [PSMNet](https://github.com/JiaRenChang/PSMNet) to finetune the pretrained models on KITTI datasets and generate disparity maps for submission. Use *finetune.sh* and *submission.sh* to do them respectively.

### Pretrained Model
We will release the pretrained models used in the paper soon.

## Results
### Results on Scene Flow dataset

| Model | EPE | Memory (GB) | Runtime (ms) on Tesla V100 |
|---|---|---|---|
| FADNet | 0.83 | 3.87 | 48.1 |
| [DispNetC](https://arxiv.org/pdf/1512.02134) | 1.68 | 1.62 | 18.7 |
| [PSMNet](https://arxiv.org/abs/1803.08669) | 1.09 | 13.99 | 399.3 |
| [GANet](https://arxiv.org/abs/1904.06587) | 0.84 | 29.1 | 2251.1 |

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
