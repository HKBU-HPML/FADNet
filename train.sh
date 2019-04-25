net=dispnetcres
loss=loss_configs/dispnetcres_flying_1.json
lr=0.001
outf_model=models/dispCSRes-1
logf=logs/dispCSRes-1.log
devices=0,1,2,3
trainlist=lists/SHAOHUAI_CLEAN_FlyingThings3D_release_TRAIN.list
vallist=lists/SHAOHUAI_CLEAN_FlyingThings3D_release_TEST.list
startR=0
startE=5
endE=20
batchSize=32
#model=none
model=models/dispCSRes-1/dispnetcres_0_4.pth
#model=models/dispnet-corr1d-best.pth
python main.py --cuda --net $net --loss $loss --lr $lr \
               --outf $outf_model --logFile $logf \
               --devices $devices --batch_size $batchSize \
               --trainlist $trainlist --vallist $vallist \
               --startRound $startR --startEpoch $startE --endEpoch $endE \
               --model $model

