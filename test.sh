net="${net:-fadnet}"
maxdisp=-1
#net=psmnet
#maxdisp=192
datapath=/datasets
dataset=sceneflow
trainlist=lists/SceneFlow.list
vallist=lists/FlyingThings3D_release_TEST.list
#vallist=lists/nano_test.list

loss=loss_configs/test.json
outf_model=models/test/
logf=logs/${net}_test_on_${dataset}-n32.log

lr=1e-4
#devices=0,1,2,3
devices=0
startR=0
startE=0
batchSize=1
#model=models/fadnet.pth
model=models/fadnet-sceneflow-n32sf/fadnet_3_27.pth
#model=models/mobilefadnet-sceneflow-v4/model_best.pth
python main.py --cuda --net $net --loss $loss --lr $lr \
               --outf $outf_model --logFile $logf \
               --devices $devices --batch_size $batchSize \
               --datapath $datapath \
               --trainlist $trainlist --vallist $vallist \
               --dataset $dataset --maxdisp $maxdisp \
               --startRound $startR --startEpoch $startE \
               --model $model 

