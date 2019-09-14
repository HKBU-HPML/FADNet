#model=models/dispnet-corr1d-best.pth
model=/mnt/sdc1/blackjack/data/dispnet-corr1d-best.pth
outf=detect_results/dispnetcres-flying-test
filelist=lists/SHAOHUAI_CLEAN_FlyingThings3D_release_TEST.list
filepath=data
net=psmnet

CUDA_VISIBLE_DEVICES=0 python detecter.py --model $model --rp $outf --filelist $filelist --filepath $filepath --net $net --batchSize 1
