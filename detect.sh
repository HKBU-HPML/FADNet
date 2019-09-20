#net=psmnet
#model=models/psmnet/model_best.pth

dataset=driving
net=dispnetcres

model=models/dispnetcres/fadnet.pth
outf=detect_results/fadnet-${dataset}/
#filelist=lists/SHAOHUAI_CLEAN_FlyingThings3D_release_TEST.list
filelist=lists/${dataset}_release.list
filepath=data

CUDA_VISIBLE_DEVICES=0,1 python detecter.py --model $model --rp $outf --filelist $filelist --filepath $filepath --devices 0,1 --net ${net}
