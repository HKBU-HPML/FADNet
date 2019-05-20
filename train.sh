net=dispnetcres
loss=loss_configs/dispnetcres_flying.json
outf_model=models/dispCSRes
logf=logs/dispCSRes.log
#net=multicorrnet
#loss=loss_configs/dispnetcres_flying.json
#outf_model=models/multicorrnet
#logf=logs/multicorrnet.log
#net=dispnetc
#loss=loss_configs/dispnetcres_flying.json
#outf_model=models/dispnetc-corr20-regression
#logf=logs/dispnetc-corr20-regression.log

lr=1e-4
#trainlist=lists/SHAOHUAI_CLEAN_FlyingThings3D_release_TRAIN.list
#vallist=lists/SHAOHUAI_CLEAN_FlyingThings3D_release_TEST.list
trainlist=lists/FlyingThings3D_release_TRAIN.list
vallist=lists/FlyingThings3D_release_TEST.list
devices=0,1
startR=0
startE=0
endE=50
batchSize=8
model=none
#model=models/dispCSRes-1/dispnetcres_0_19.pth
#model=models/dispCSRes/model_best.pth
#model=models/dispnetc/dispnetc_0_9.pth
#model=models/model_best.pth
#model=models/dispCSRes-regression/dispnetcres_1_16.pth
#model=models/multicorrnet/multicorrnet_0_3.pth
python main.py --cuda --net $net --loss $loss --lr $lr \
               --outf $outf_model --logFile $logf \
               --devices $devices --batch_size $batchSize \
               --trainlist $trainlist --vallist $vallist \
               --startRound $startR --startEpoch $startE --endEpoch $endE \
               --model $model

