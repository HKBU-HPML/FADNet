#python detecter.py --model ./cleandata-models-dispC-resnet/dispS_epoch_8.pth --rp ./detect_result_cd --filepath /home/datasets/imagenet/
#python detecter.py --model ./dispCSRes-model_best.pth --rp ./detect_result_csr --filepath /home/datasets/imagenet/ --filelist CLEAN_FlyingThings3D_release_TEST.list
# python detecter.py --model model_best.pth --rp ./detect_result --filepath /home/datasets/imagenet/

# detect with dispCSRes
#python detecter.py --model ./cleandata-models-dispCSRes/model_best.pth --rp ./detect_result_crop --filepath /home/datasets/imagenet

python detecter.py --model ./cc-cleandata-dispCSRes-model_best.pth --rp ./cc_detect_result --filepath /home/datasets/imagenet
# python detecter.py --model ./cleandata-dispCSRes-model_best.pth --rp ./detect_result_l2norm --filepath /home/datasets/imagenet

<<<<<<< HEAD
#python detecter.py --model ./models/cc-dispCSRes-model_best.pth --rp ./cc-detect_result_l2norm --filepath /home/datasets/imagenet --devices 1,2 --batchSize 2 
=======
# python detecter.py --model ./models/cc-dispCSRes-model_best.pth --rp ./cc-detect_result_l2norm --filepath /home/datasets/imagenet --devices 1,2 --batchSize 2 
# detect with titanx
python detecter.py --model ./models/cc-dispCSRes-model_best.pth --rp ./cc-detect_result_l2norm --filepath data 
>>>>>>> 51886211043f867a82e704a1c68e1d8aa727be2f

# python detecter.py --model ./models/cc-dispCSRes-model_best.pth --rp ./cc-detect_result_l2norm --filepath /home/datasets/imagenet

