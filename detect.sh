#net=psmnet
#model=models/psmnet/model_best.pth

dataset=SIRS_metal
net=dispnetcres

#model=models/dispnetcres/model_best.pth
model=models/fadnet_sf.pth
outf=detect_results/dispnetcres-${dataset}/
#filelist=lists/SHAOHUAI_CLEAN_FlyingThings3D_release_TEST.list
filelist=lists/${dataset}_test.list
filepath=data

CUDA_VISIBLE_DEVICES=0,1 python detecter.py --model $model --rp $outf --filelist $filelist --filepath $filepath --devices 0,1 --net ${net} --disp-on
