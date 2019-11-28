#net=psmnet
#model=models/psmnet/model_best.pth

dataset=sceneflow
net=dtonnet
#net=dispnetcres

#model=models/dispnetcres/model_best.pth
#model=models/fadnet_sf.pth
#model=models/dispnormnet-test/model_best.pth

model=data/cvpr2020/models/dtonnet-sf-fl1050-d0.921-n15.6.pth
#model=data/cvpr2020/models/psmnet.pth
#model=data/cvpr2020/models/dispnetc.pth

outf=data/cvpr2020/results/${net}-${dataset}/
#outf=data/cvpr2020/results/psmnet-${dataset}/
#filelist=lists/SHAOHUAI_CLEAN_FlyingThings3D_release_TEST.list
#filelist=lists/${dataset}_test.list
#filelist=lists/FlyingThings3D_release_TEST_norm.list
filelist=lists/pcd_sample_data.list
filepath=data

CUDA_VISIBLE_DEVICES=2 python detecter.py --model $model --rp $outf --filelist $filelist --filepath $filepath --devices 0 --net ${net} --disp-on --norm-on
