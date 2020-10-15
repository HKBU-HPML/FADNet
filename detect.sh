net=mobilefadnet
dataset=sceneflow

model=models/mobilefadnet-sceneflow/model_best.pth
outf=detect_results/${net}-${dataset}/

filelist=lists/FlyingThings3D_release_TEST.list
#filelist=lists/nano_fake.list
filepath=/datasets
#filepath=./data

#CUDA_VISIBLE_DEVICES=0 python detecter.py --model $model --rp $outf --filelist $filelist --filepath $filepath --devices 0 --net ${net} 
CUDA_VISIBLE_DEVICES=0 python detecter_rt.py --model $model --rp $outf --filelist $filelist --filepath $filepath --devices 0 --net ${net} 
