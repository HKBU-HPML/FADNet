#net=psmnet
#model=models/psmnet/model_best.pth

dataset=sceneflow
net=dispnormnet

#model=models/dispnetcres/model_best.pth
#model=models/fadnet_sf.pth
model=models/dispnormnet-test/model_best.pth
outf=detect_results/${net}-${dataset}/
#filelist=lists/SHAOHUAI_CLEAN_FlyingThings3D_release_TEST.list
#filelist=lists/${dataset}_test.list
filelist=lists/FlyingThings3D_release_TEST_norm.list
filepath=data

CUDA_VISIBLE_DEVICES=2 python detecter.py --model $model --rp $outf --filelist $filelist --filepath $filepath --devices 0 --net ${net} --disp-on --norm-on
