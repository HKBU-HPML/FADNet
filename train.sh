net=dispnetcres
loss=loss_configs/dispnetcres_flying.json
lr=1e-4
outf_model=models/dispCSRes-regression
logf=logs/dispCSRes-regression.log
devices=0,1,2,3
#trainlist=lists/SHAOHUAI_CLEAN_FlyingThings3D_release_TRAIN.list
#vallist=lists/SHAOHUAI_CLEAN_FlyingThings3D_release_TEST.list
trainlist=lists/FlyingThings3D_release_TRAIN.list
vallist=lists/FlyingThings3D_release_TEST.list
startR=0
startE=0
endE=60
batchSize=8
model=none
#model=models/dispCSRes-1/dispnetcres_0_19.pth
#model=models/dispCSRes/model_best.pth
#model=models/model_best.pth
python main.py --cuda --net $net --loss $loss --lr $lr \
               --outf $outf_model --logFile $logf \
               --devices $devices --batch_size $batchSize \
               --trainlist $trainlist --vallist $vallist \
               --startRound $startR --startEpoch $startE --endEpoch $endE \
               --model $model

