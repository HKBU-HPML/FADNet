#python detecter.py --model ./cleandata-models-dispC-resnet/dispS_epoch_8.pth --rp ./detect_result_cd --filepath /home/datasets/imagenet/
#python detecter.py --model ./dispCSRes-model_best.pth --rp ./detect_result_csr --filepath /home/datasets/imagenet/ --filelist CLEAN_FlyingThings3D_release_TEST.list
# python detecter.py --model model_best.pth --rp ./detect_result --filepath /home/datasets/imagenet/

# detect with dispCSRes
#python detecter.py --model ./cleandata-models-dispCSRes/model_best.pth --rp ./detect_result_crop --filepath /home/datasets/imagenet

python detecter.py --model ./cleandata-dispCSRes-model_best.pth --rp ./detect_result_l2norm --filepath /home/datasets/imagenet
