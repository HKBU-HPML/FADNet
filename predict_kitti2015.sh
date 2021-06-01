PY=/home/esetstore/fadnet/bin/python
CUDA_VISIBLE_DEVICES=1 $PY predict.py \
                --kitti2015=1    --maxdisp=192 \
                --crop_height=384  --crop_width=1280  \
                --datapath='./data/' \
                --list='./lists/kitti2015_test.list' \
                --savepath='./predict/kitti2015/' \
                --loadmodel './models/fadnet-ft-rvc-1e-4/model_best.pth' 

