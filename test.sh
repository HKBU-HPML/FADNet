#net=dispnetcres
#net=dispnetcss
net=dispnormnet
#loss=loss_configs/dispnetcres_flying.json
#outf_model=models/dispCSRes-regression
#logf=logs/dispCSRes-regression.log
#net=multicorrnet
#loss=loss_configs/dispnetcres_flying.json
#outf_model=models/multicorrnet
#logf=logs/multicorrnet.log

#net=normnets
label=sceneflow
#label=over_exposure
#label=dark
#label=glass_mirror
#label=len_flare
#label=metal
loss=loss_configs/dispnetcss_one.json
#outf_model=models/dispnetc_ue_norm2
#logf=logs/dispnetc_norm.log

outf_model=models/dispnetcss_test
#logf=logs/dispnetcss_SIRS.log
#logf=logs/dispnetcss_fly_on_SIRS.log
#logf=logs/dispnetcss_sceneflow_on_${label}.log
logf=logs/dispnormnet_SIRS_on_${label}.log

lr=1e-4
devices=0
#trainlist=lists/SHAOHUAI_CLEAN_FlyingThings3D_release_TRAIN.list
#vallist=lists/SHAOHUAI_CLEAN_FlyingThings3D_release_TEST.list
#trainlist=lists/home_train.list
#trainlist=lists/ue_sperate_train.list
#trainlist=lists/home_extend_train.list
#trainlist=lists/FlyingThings3D_release_TRAIN.list
#vallist=lists/ue_sperate_test.list
#vallist=lists/office_test.list
#vallist=lists/home_extend_test.list
#vallist=lists/home_test.list
#vallist=lists/KITTI_TRAIN.list

trainlist=lists/SIRSDataset_train.list
#vallist=lists/SIRS_${label}_test.list
vallist=lists/FlyingThings3D_release_TEST.list

#load_norm=True


startR=0
startE=0
endE=0
batchSize=8
#model=none
#model=models/dispnetcss_SIRS/dispnetcss_0_0.pth
#model=models/dispnetcss_SIRS/model_best.pth
#model=models/dispnetcss_sceneflow/model_best.pth
model=models/dispnormnet_SIRS-mse/model_best.pth
#model=models/dispnetc_ue4_sperate.pth
#model=models/dispnetc_fly0.pth
#model=models/dispnetc_ue4_total2.pth
#model=models/dispnetc_texture_alter.pth
#model=models/dispnetc_ue_mix.pth
#model=models/dispCSRes-1/dispnetcres_0_19.pth
#model=models/dispCSRes/model_best.pth
#model=models/model_best.pth
#model=models/dispCSRes-regression/dispnetcres_1_16.pth
#model=models/multicorrnet/multicorrnet_0_3.pth
#model=models/dispnetc_ue_norm2/model_best.pth
#CUDA_VISIBLE_DEVICES=1 
python main.py --cuda --net $net --loss $loss --lr $lr \
               --outf $outf_model --logFile $logf \
               --devices $devices --batch_size $batchSize \
               --trainlist $trainlist --vallist $vallist \
               --startRound $startR --startEpoch $startE --endEpoch $endE \
               --model $model #\
	       #--load_norm $load_norm 

