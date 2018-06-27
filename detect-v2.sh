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
python detecter.py --model ./models/virtual-style-dispCSR/model_best.pth --rp ./detect_results/virtual-style-virtual --filelist ./lists/girl_style.list --filepath / 

# moto 
#python detecter.py --model ./models/girl-models-dispCSRes-finetune-crop-3nd/dispS_epoch_40.pth --rp ./detect_result/moto --filepath /home/datasets/imagenet/dispnet --filelist lists/moto.list
