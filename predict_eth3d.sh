PY=/home/esetstore/fadnet/bin/python
CUDA_VISIBLE_DEVICES=1 $PY predict.py \
                --eth3d=1    --maxdisp=192 \
                --crop_height=576  --crop_width=960  \
                --datapath='./data/' \
                --list='./lists/eth3d_test.list' \
                --savepath='./predict/eth3d/' \
                --loadmodel './models/fadnet-ft-rvc-1e-4/model_best.pth' 

