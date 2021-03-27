net="${net:-fadnet}"
maxdisp=-1
#net=psmnet
#maxdisp=192
datapath=/home/datasets
#dataset=sceneflow
#trainlist=lists/SceneFlow.list
#vallist=lists/FlyingThings3D_release_TEST.list
dataset=irs
trainlist=lists/IRS_TRAIN.list
vallist=lists/IRS_len_flare_test.list
#vallist=lists/flying_short.list

loss=loss_configs/test.json
outf_model=models/test/
#logf=logs/${net}_test_on_${dataset}.log
logf=logs/${net}_ft3d+irs_len_flare.log

lr=1e-4
devices=0,1,2,3
startR=0
startE=0
batchSize=32
model="${model:-models/fadnet-irs/model_best.pth}"
#model=models/mobilefadnet-sceneflow-v3/model_best.pth
python main.py --cuda --net $net --loss $loss --lr $lr \
               --outf $outf_model --logFile $logf \
               --devices $devices --batch_size $batchSize \
               --datapath $datapath \
               --trainlist $trainlist --vallist $vallist \
               --dataset $dataset --maxdisp $maxdisp \
               --startRound $startR --startEpoch $startE \
               --model $model 

