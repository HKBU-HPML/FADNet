PY=/home/esetstore/fadnet/bin/python
CUDA_VISIBLE_DEVICES=1 $PY predict.py \
                --middlebury=1    --maxdisp=360 \
                --crop_height=1024  --crop_width=1536  \
                --datapath='./data/MiddEval3/' \
                --list='./lists/middeval3q_test.list' \
                --savepath='./predict/middlebury/' \
                --loadmodel ./models/fadnet-sceneflow.pth
                #--loadmodel './models/fadnet-ft-rvc-16e-4-mdq/model_best.pth' 
                #--loadmodel '/home/esetstore/repos/FADNet/models/fadnet-ft-rvc-16e-4-largemd/model_best.pth' 

