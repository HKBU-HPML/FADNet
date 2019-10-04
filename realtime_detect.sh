net=dispnormnet
model=models/dispnormnet_resblock_model_best.pth

CUDA_VISIBLE_DEVICES=0 python realtime_detect.py --model $model --devices 0 --net ${net}
