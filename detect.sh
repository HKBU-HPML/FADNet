#################### detecter v1 ####################
#python detecter.py --model ./cleandata-models-dispC-resnet/dispS_epoch_8.pth --rp ./detect_result_cd --filepath /home/datasets/imagenet/
#python detecter.py --model ./models/girl-models-dispCSRes-finetune-crop-3nd/dispS_epoch_40.pth --rp ./detect_result/flyingthings --filepath /home/datasets/imagenet/ --filelist ./lists/CLEAN_FlyingThings3D_release_TEST.list
#python detecter.py --model ./cleandata-dispCSRes-model_best.pth --rp ./detect_result/flyingthings-original --filepath /home/datasets/imagenet/ --filelist ./lists/CLEAN_FlyingThings3D_release_TEST.list

# python detecter.py --model model_best.pth --rp ./detect_result --filepath /home/datasets/imagenet/

# detect with dispCSRes
# python detecter.py --model ./cleandata-models-dispCSRes/model_best.pth --rp ./detect_result_crop --filepath /home/datasets/imagenet

#python detecter.py --model ./cc-cleandata-dispCSRes-model_best.pth --rp ./cc_detect_result --filepath /home/datasets/imagenet
#python detecter.py --model ./models/cleandata-dispCSRes-model_best.pth --rp ./flying_detect_cleandata --filepath /home/datasets/imagenet --filelist FlyingThings3D_release_TEST.list
#python detecter.py --model ./cc-models-dispC-resnet-cleandata-mix/model_best.pth --rp ./flying_detect_occlusion --filepath /home/datasets/imagenet --filelist RB_FlyingThings3D_release_TEST.list
#python detecter.py --model ./models/cleandata-models-dispCSR/model_best.pth --rp ./detect_results/flying_detect_cleandata --filepath /home/datasets/imagenet --filelist lists/FlyingThings3D_release_TEST.list

python detecter.py --model ./models/finetune-from-flyingthings-3nd/dispS_epoch_12.pth --rp ./detect_result/flyingthings_model --filepath /home/datasets/imagenet/dispnet --filelist lists/real_release.list #lists/girl20_TEST.list #--filelist 
#python detecter.py --model ./models/finetune-from-flyingthings-2nd/dispS_epoch_59.pth --rp ./detect_result/flyingthings_model --filepath /home/datasets/imagenet/dispnet --filelist lists/custom.list #lists/girl20_TEST.list #--filelist 

# python detecter.py --model ./models/cc-dispCSRes-model_best.pth --rp ./cc-detect_result_l2norm --filepath /home/datasets/imagenet --devices 1,2 --batchSize 2 

# FlyingThings
#python detecter.py --model ./models/cc-dispCSRes-model_best.pth --rp ./cc_detect_result_l2norm --filepath data 
# python detecter.py --model ./cleandata-models-dispC-resnet-relu-ft-rb/model_best.pth --rp ./flying_detect_result --filepath data --filelist RB_FlyingThings3D_release_TEST.list

# real camera
python detecter.py --model ./models/real-dispCSR/model_best.pth --rp ./detect_results/real_detect_virtual --filepath /home/datasets/imagenet/dispnet --filelist lists/test.list
#python detecter.py --model ./models/girl-models-dispCSRes-finetune-crop-3nd/model_best.pth --rp ./detect_result/real_detect_cleandata_cropscale --filepath /home/datasets/imagenet/dispnet --filelist lists/real_release.list
#python detecter.py --model ./models/girl02-train-dispCSRes-crop-3nd/model_best.pth --rp ./detect_result/real_detect_cleandata_cropscale --filepath /home/datasets/imagenet/dispnet --filelist lists/real_release.list
#python detecter.py --model /home/comp/csshshi/fromsz/model_best.pth --rp ./detect_result/cropfrom1024model_real_detect_cleandata_scale1024 --filepath /home/datasets/imagenet/dispnet --filelist lists/real_release.list
#python detecter.py --model ./models/girl02-train-dispCSR-crop1024-3nd/dispS_epoch_99.pth --rp ./detect_result/cropfrom1024model_real_detect_cleandata_scale1024 --filepath /home/datasets/imagenet/dispnet --filelist lists/real_release.list #lists/girl20_TEST.list #--filelist 
# python detecter.py --model ./cleandata-models-dispC-resnet-relu-ft-rb-mix/model_best.pth --rp ./real_detect_result_s1.5 --filepath data/dispnet --filelist real_sgm_release.list
# detect with titanx
# python detecter.py --model ./models/cc-dispCSRes-model_best.pth --rp ./cc-detect_result_l2norm --filepath data 

# python detecter.py --model ./models/cc-dispCSRes-model_best.pth --rp ./cc-detect_result_l2norm --filepath /home/datasets/imagenet

# girl data
#python detecter.py --model ./girl-models-dispCSRes-finetune-changeweight/model_best.pth --rp ./girl_detect_cleandata --filepath /home/datasets/imagenet/dispnet/virtual/girl --filelist ./lists/girl.list

# girl02 data
#python detecter.py --model ./models/girl-models-dispCSRes-finetune-crop-3nd/dispS_epoch_40.pth --rp ./detect_result/girl-crop-noscale-3nd-epe0.152 --filepath /home/datasets/imagenet/dispnet/virtual --filelist ./lists/girl05.list
#python detecter.py --model ./models/cleandata-models-dispCSR-girl/model_best.pth --rp ./detect_results/girl_detect_cleandata --filepath ./data/ --filelist lists/girl_TEST.list
#python detecter.py --model ./models/cleandata-models-dispCSR-girl/model_best.pth --rp ./detect_results/girl_detect_cleandata --filepath /home/datasets/imagenet/dispnet/virtual --filelist lists/girl_TEST.list
#python detecter.py --model ./models/girl-models-dispCSRes-finetune-crop-3nd/model_best.pth --rp ./detect_result/girl-crop-scale512 --filepath /home/datasets/imagenet/dispnet/virtual --filelist ./lists/girl09.list

# moto 
#python detecter.py --model ./models/girl-models-dispCSRes-finetune-crop-3nd/dispS_epoch_40.pth --rp ./detect_result/moto --filepath /home/datasets/imagenet/dispnet --filelist lists/moto.list

#################### detecter v2 ####################
#python detecter.py --model ./cleandata-models-dispC-resnet/dispS_epoch_8.pth --rp ./detect_result_cd --filepath /home/datasets/imagenet/
#python detecter.py --model ./models/girl-models-dispCSRes-finetune-crop-3nd/dispS_epoch_40.pth --rp ./detect_result/flyingthings --filepath /home/datasets/imagenet/ --filelist ./lists/CLEAN_FlyingThings3D_release_TEST.list
#python detecter.py --model ./cleandata-dispCSRes-model_best.pth --rp ./detect_result/flyingthings-original --filepath /home/datasets/imagenet/ --filelist ./lists/CLEAN_FlyingThings3D_release_TEST.list

# python detecter.py --model model_best.pth --rp ./detect_result --filepath /home/datasets/imagenet/

# detect with dispCSRes
# python detecter.py --model ./cleandata-models-dispCSRes/model_best.pth --rp ./detect_result_crop --filepath /home/datasets/imagenet

#python detecter.py --model ./cc-cleandata-dispCSRes-model_best.pth --rp ./cc_detect_result --filepath /home/datasets/imagenet
#python detecter.py --model ./models/cleandata-dispCSRes-model_best.pth --rp ./flying_detect_cleandata --filepath /home/datasets/imagenet --filelist FlyingThings3D_release_TEST.list
#python detecter.py --model ./cc-models-dispC-resnet-cleandata-mix/model_best.pth --rp ./flying_detect_occlusion --filepath /home/datasets/imagenet --filelist RB_FlyingThings3D_release_TEST.list
#python detecter.py --model ./models/cleandata-models-dispCSR/model_best.pth --rp ./detect_results/flying_detect_cleandata --filepath /home/datasets/imagenet --filelist lists/FlyingThings3D_release_TEST.list
# python detecter.py --model ./cleandata-models-dispCSRes-exp/model_best.pth --rp ./real_detect_cleandata --filepath ./data --filelist real_release.list

# python detecter.py --model ./models/cc-dispCSRes-model_best.pth --rp ./cc-detect_result_l2norm --filepath /home/datasets/imagenet --devices 1,2 --batchSize 2 

# FlyingThings
#python detecter.py --model ./models/cc-dispCSRes-model_best.pth --rp ./cc_detect_result_l2norm --filepath data 
# python detecter.py --model ./cleandata-models-dispC-resnet-relu-ft-rb/model_best.pth --rp ./flying_detect_result --filepath data --filelist RB_FlyingThings3D_release_TEST.list

# real camera
#python detecter.py --model ./models/girl-models-dispCSRes-finetune-crop-3nd/model_best.pth --rp ./detect_result/real_detect_cleandata --filepath /home/datasets/imagenet/dispnet --filelist lists/test.list
# python detecter.py --model ./cleandata-models-dispC-resnet-relu-ft-rb-mix/model_best.pth --rp ./real_detect_result_s1.5 --filepath data/dispnet --filelist real_sgm_release.list
# detect with titanx
# python detecter.py --model ./models/cc-dispCSRes-model_best.pth --rp ./cc-detect_result_l2norm --filepath data 
# python detecter.py --model ./models/virtual-style-dispCSR/model_best.pth --rp ./detect_results/virtual-style-real --filelist ./lists/real_style.list --filepath /data2 

# python detecter.py --model ./models/cc-dispCSRes-model_best.pth --rp ./cc-detect_result_l2norm --filepath /home/datasets/imagenet

# girl data
#python detecter.py --model ./girl-models-dispCSRes-finetune-changeweight/model_best.pth --rp ./girl_detect_cleandata --filepath /home/datasets/imagenet/dispnet/virtual/girl --filelist ./lists/girl.list

# girl style data
#python detecter.py --model ./models/girl-models-dispCSRes-finetune-crop-3nd/dispS_epoch_40.pth --rp ./detect_result/girl-crop-noscale-3nd-epe0.152 --filepath /home/datasets/imagenet/dispnet/virtual --filelist ./lists/girl05.list
#CUDA_VISIBLE_DEVICES=1 python detecter.py --model ./models/real-dispCSR/model_best.pth --rp ./detect_results/real --filelist ./lists/real_release.list --filepath /media/external/data/virtual
#CUDA_VISIBLE_DEVICES=1 python detecter.py --model ./models/cleandata-models-dispCSRes-exp/model_best.pth --rp ./detect_results/real --filelist ./lists/real_release.list --filepath /media/external/data/virtual
CUDA_VISIBLE_DEVICES=0 python detecter.py --model ./models/dispCSRes-corr-20-r4/model_best.pth --rp ./detect_results/real-dispnet3 --filelist ./lists/real_release.list --filepath /home/vradmin/dispflownet-release/data
#CUDA_VISIBLE_DEVICES=1 python detecter.py --model ./models/dispCSRes-corr-20-r4/model_best.pth --rp ./detect_results/flying-dispnet3 --filelist ./lists/SHAOHUAI_CLEAN_FlyingThings3D_release_TEST.list --filepath /home/vradmin/dispflownet-release/data
#CUDA_VISIBLE_DEVICES=1 python detecter.py --model ./models/flying-real-dispCSRWithMono-in1024-r2/dispS_epoch_43.pth --rp ./detect_results/real-monodepth --filelist ./lists/real_release.list --filepath /media/external/data/virtual
#CUDA_VISIBLE_DEVICES=1 python detecter.py --model ./models/real-dispCSR/dispS_epoch_77.pth --rp ./detect_results/real-monodepth --filelist ./lists/real_release.list --filepath /media/external/data/virtual
#CUDA_VISIBLE_DEVICES=1 python detecter.py --model ./models/real-dispCSR/model_best.pth --rp ./detect_results/real-monodepth --filelist ./lists/real_release.list --filepath /media/external/data/virtual

# moto 
#python detecter.py --model ./models/girl-models-dispCSRes-finetune-crop-3nd/dispS_epoch_40.pth --rp ./detect_result/moto --filepath /home/datasets/imagenet/dispnet --filelist lists/moto.list
#python detecter.py --model /data2/models/girl1024x1024/model_best.pth --rp /data2/models/detect_result/girl1024x1024 --filepath /data2/virtual --filelist ./lists/virtual01-1024x1024_TEST.list
#python detecter.py --model /data2/models/girl1024x1024/model_best.pth --rp /data2/models/detect_result/real1024x1024 --filepath /data2 --filelist lists/real_release.list 

#################### detecter v3 ####################
#python detecter_v3.py --model ./models/girl-domain-transfer-models-dispCSRes/model_best.pth --rp ./detect_result/domain_adaptation --filepath /home/datasets/imagenet/dispnet/virtual --filelist ./lists/girl09.list
#python detecter_v3.py --model ./models/girl-domain-transfer-models-dispCSRes/model_best.pth --rp ./detect_result/domain_adaptation --filepath /home/datasets/imagenet/dispnet/virtual --filelist ./lists/girl09_TEST.list

# TITANX
#python detecter_v3.py --model ./models/girl-domain-transfer-models-dispCSRes/model_best.pth --rp ./detect_result/domain_adaptation --filepath ~/dispflownet-release/data/girl --filelist ./lists/girl09.list
#python detecter_v3.py --model ./models/girl-domain-transfer-models-dispCSRes-finetune2/model_best.pth --rp ./detect_result/domain_adaptation-finetune2 --filepath ~/dispflownet-release/data/girl --filelist lists/real_release.list
python detecter_v3.py --model /home/comp/csshshi/fromsz/dispS_epoch_37.pth --rp ./detect_result/domain_adaptation --filepath /home/datasets/imagenet/dispnet --filelist lists/real_release.list
#python detecter_v3.py --model ./models/girl-domain-transfer-models-dispCSRes/model_best.pth --rp ./detect_result/domain_adaptation_1024 --filepath /home/datasets/imagenet/dispnet --filelist lists/real_release.list
