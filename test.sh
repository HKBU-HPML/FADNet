net=fadnet
maxdisp=-1
#net=psmnet
#maxdisp=192
dataset=sceneflow
trainlist=lists/SceneFlow.list
vallist=lists/FlyingThings3D_release_TEST.list

loss=loss_configs/test.json
outf_model=models/test/
logf=logs/${net}_test_on_${dataset}.log

lr=1e-4
devices=0
startR=0
startE=0
batchSize=8
model=models/fadnet.pth
python3 main.py --cuda --net $net --loss $loss --lr $lr \
               --outf $outf_model --logFile $logf \
               --devices $devices --batch_size $batchSize \
               --trainlist $trainlist --vallist $vallist \
               --dataset $dataset --maxdisp $maxdisp \
               --startRound $startR --startEpoch $startE \
               --model $model 

