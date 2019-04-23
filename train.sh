net=dispnetcres
loss=loss_configs/dispnetcres_flying.json
lr=0.001
outf_model=models/dispCSRes
logf=logs/dispCSRes.log
devices=0,1,2,3
trainlist=lists/SHAOHUAI_CLEAN_FlyingThings3D_release_TRAIN.list
vallist=lists/SHAOHUAI_CLEAN_FlyingThings3D_release_TEST.list
startR=0
startE=0
endE=2
batchSize=12
python main.py --cuda --net $net --loss $loss --lr $lr \
               --outf $outf_model --logFile $logf \
               --devices $devices --batch_size $batchSize \
               --trainlist $trainlist --vallist $vallist \
               --startRound $startR --startEpoch $startE --endEpoch $endE \

