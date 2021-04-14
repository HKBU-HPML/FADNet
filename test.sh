net="${net:-fadnet}"
#datapath=/datasets
#maxdisp=-1
#net=psmnet
#maxdisp=192
#dataset=sceneflow
#trainlist=lists/SceneFlow.list
#vallist=lists/FlyingThings3D_release_TEST.list
#trainlist=lists/IRS_TRAIN.list
#vallist=lists/IRS_TEST.list
maxdisp=-1
#dataset=sintel
#trainlist=lists/Sintel_ALL.list
#vallist=lists/Sintel_ALL.list
datapath=/datasets/kitti2015/training/
dataset=kitti2015
trainlist=KITTI_TRAIN.list
vallist=KITTI_TEST.list

loss=loss_configs/test.json
outf_model=models/test/
logf=logs/${net}_test_on_${dataset}.log
#logf=logs/${net}_ft3d+irs_len_flare.log

lr=1e-4
devices=0
startR=0
startE=0
batchSize=1
model="${model:-models/fadnet-sceneflow/model_best.pth}"
python main.py --cuda --net $net --loss $loss --lr $lr \
               --outf $outf_model --logFile $logf \
               --devices $devices --batch_size $batchSize \
               --datapath $datapath \
               --trainlist $trainlist --vallist $vallist \
               --dataset $dataset --maxdisp $maxdisp \
               --startRound $startR --startEpoch $startE \
               --model $model 

