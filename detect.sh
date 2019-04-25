model=models/dispnet-corr1d-best.pth
outf=detect_results/dispnetcres-flying-test
filelist=lists/SHAOHUAI_CLEAN_FlyingThings3D_release_TEST.list
filepath=data

CUDA_VISIBLE_DEVICES=0 python detecter.py --model $model --rp $outf --filelist $filelist --filepath $filepath
