net=fadnet
dataset=sceneflow

model=models/fadnet.pth
outf=detect_results/${net}-${dataset}/

#filelist=lists/FlyingThings3D_release_TEST.list
filelist=lists/nano_fake.list
filepath=data

CUDA_VISIBLE_DEVICES=0 python3 detecter.py --model $model --rp $outf --filelist $filelist --filepath $filepath --devices 0 --net ${net} 
