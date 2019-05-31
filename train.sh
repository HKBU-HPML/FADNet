#!/usr/bin/bash
net=dispnetcres
loss=loss_configs/dispnetcres_flying.json
outf_model=/datasets/sh_kittis/models/ShDispCSRes
logf=logs/dispCSRes.log
#net=multicorrnet
#loss=loss_configs/dispnetc_flying.json
#outf_model=models/multicorrnet
#logf=logs/multicorrnet.log
#net=dispnetc
#loss=loss_configs/dispnetc_flying.json
#outf_model=models/dispnetc
#logf=logs/dispnetc.log

lr=1e-4
#trainlist=lists/SHAOHUAI_CLEAN_FlyingThings3D_release_TRAIN.list
#vallist=lists/SHAOHUAI_CLEAN_FlyingThings3D_release_TEST.list
#trainlist=lists/FlyingThings3D_release_TRAIN.list
trainlist=lists/SceneFlow.list
vallist=lists/FlyingThings3D_release_TEST.list
devices=0,1,2,3
startR=1
startE=1
endE=50
batchSize=32
#model=none
#model=models/dispCSRes/dispnetcres_1_29.pth
#model=/datasets/sh_kittis/models/ShDispCSRes/model_best.pth
#model=models/dispnetc/dispnetc_0_9.pth
#model=models/model_best.pth
#model=models/dispCSRes-regression/dispnetcres_1_16.pth
#model=models/multicorrnet/multicorrnet_0_3.pth
python main.py --cuda --net $net --loss $loss --lr $lr \
               --outf $outf_model --logFile $logf \
               --devices $devices --batch_size $batchSize \
               --trainlist $trainlist --vallist $vallist \
               --startRound $startR --startEpoch $startE --endEpoch $endE \
               --datapath /datasets 
#               --model $model

