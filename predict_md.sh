PY=/home/esetstore/fadnet/bin/python
CUDA_VISIBLE_DEVICES=1 $PY predict.py \
                --middlebury=1    --maxdisp=360 \
                --crop_height=1024  --crop_width=1536  \
                --datapath='./data/MiddEval3/' \
                --list='./lists/middeval3h_test.list' \
                --savepath='./predict/middlebury/' \
                --loadmodel './models/fadnet-ft-rvc-1e-4/model_best.pth' 
                #--loadmodel './models/fadnet-sceneflow-dynamic.pth' 

