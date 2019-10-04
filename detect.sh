#net=psmnet
#model=models/psmnet/model_best.pth

dataset=driving
net=dispnormnet

model=models/dispnormnet_resblock_model_best.pth
outf=detect_results/dark_valid_norm/
filelist=lists/ue_dark_valid.list
#filelist=lists/${dataset}_release.list
filepath=data

CUDA_VISIBLE_DEVICES=0 python detecter.py --rp $outf --model $model --filelist $filelist --filepath $filepath --devices 0 --net ${net}
