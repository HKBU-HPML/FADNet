#net=mobilefadnet
net=fadnet
dataset=sceneflow

#model=models/${net}-sceneflow-v3/model_best.pth
model=models/${net}-sceneflow.pth
outf=detect_results/${net}-${dataset}/
#outf=detect_results/${net}-${dataset}/

filelist=lists/FlyingThings3D_release_TEST.list
#filepath=/datasets
filepath=./data

CUDA_VISIBLE_DEVICES=0 python detecter.py --model $model --rp $outf --filelist $filelist --filepath $filepath --devices 0 --net ${net} --format pfm
