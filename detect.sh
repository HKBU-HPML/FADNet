#python detecter.py --model ./cleandata-models-dispC-resnet/dispS_epoch_8.pth --rp ./detect_result_cd --filepath /home/datasets/imagenet/
python detecter.py --model ./dispCSRes-model_best.pth --rp ./detect_result_csr --filepath /home/datasets/imagenet/ --filelist CLEAN_FlyingThings3D_release_TEST.list
