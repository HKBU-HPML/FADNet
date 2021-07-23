# dnn="${dnn:-fadnet}"
# source exp_configs/fadnet.conf

net=fadnet
loss=loss_configs/fadnet_sceneflow.json
outf_model=models/${net}-sceneflow
logf=logs/${net}-sceneflow.log

lr=2e-4
devices=0,1
dataset=sceneflow
datapath=/spyder/sceneflow
trainlist=lists/SceneFlow.list
vallist=lists/FlyingThings3D_release_TEST.list
startR=0
startE=20
batchSize=8
maxdisp=-1
model=./models/fadnet-sceneflow/fadnet_0_19_1.195.pth 

python -W ignore main.py --cuda --net $net --loss $loss --lr $lr \
               --outf $outf_model --logFile $logf \
               --devices $devices --batch_size $batchSize \
               --datapath $datapath \
               --dataset $dataset --trainlist $trainlist --vallist $vallist \
               --startRound $startR --startEpoch $startE \
               --model $model \
               --maxdisp $maxdisp \
	       --manualSeed 1024 \
