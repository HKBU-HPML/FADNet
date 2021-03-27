net="${net:-fadnet}"
#maxdisp=-1
#net=psmnet
#maxdisp=192
datapath=/home/datasets
#dataset=sceneflow
#trainlist=lists/SceneFlow.list
#vallist=lists/FlyingThings3D_release_TEST.list
dataset=irs
trainlist=lists/IRS_TRAIN.list
<<<<<<< HEAD
vallist=lists/IRS_len_flare_test.list
=======
vallist=lists/IRS_TEST.list
maxdisp=192
datapath=/datasets
dataset=sintel
trainlist=lists/Sintel_ALL.list
vallist=lists/Sintel_ALL.list
>>>>>>> e2719f7527df203c49dc162fde243dc9a1ccedc5
#vallist=lists/flying_short.list

loss=loss_configs/test.json
outf_model=models/test/
<<<<<<< HEAD
#logf=logs/${net}_test_on_${dataset}.log
logf=logs/${net}_ft3d+irs_len_flare.log
=======
logf=logs/${net}_test_on_${dataset}.log
>>>>>>> e2719f7527df203c49dc162fde243dc9a1ccedc5

lr=1e-4
devices=0
startR=0
startE=0
batchSize=32
model="${model:-models/fadnet-sceneflow/model_best.pth}"
python main.py --cuda --net $net --loss $loss --lr $lr \
               --outf $outf_model --logFile $logf \
               --devices $devices --batch_size $batchSize \
               --datapath $datapath \
               --trainlist $trainlist --vallist $vallist \
               --dataset $dataset --maxdisp $maxdisp \
               --startRound $startR --startEpoch $startE \
               --model $model 

