#net=psmnet
#model=models/psmnet/model_best.pth
net=dispnetcres
model=models/dispnetcres-sl/dispnetcres_2_8.pth
outf=detect_results/psmnet-flying/
filelist=lists/SHAOHUAI_CLEAN_FlyingThings3D_release_TEST.list
filepath=data

CUDA_VISIBLE_DEVICES=0,1 python detecter.py --model $model --rp $outf --filelist $filelist --filepath $filepath --devices 0,1 --net ${net}
