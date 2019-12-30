net=psmnet
dataset=sceneflow

model=models/psmnet.pth
outf=detect_results/${net}-${dataset}/

filelist=lists/FlyingThings3D_release_TEST.list
filepath=data

CUDA_VISIBLE_DEVICES=0 python detecter.py --model $model --rp $outf --filelist $filelist --filepath $filepath --devices 0 --net ${net} 
