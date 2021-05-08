CUDA_VISIBLE_DEVICES=0 python predict.py \
                --middlebury=1    --maxdisp=408 \
                --crop_height=1024  --crop_width=1536  \
                --datapath='./data/MiddEval3/' \
                --list='./lists/middeval3_test.list' \
                --savepath='./predict/middlebury/' \
                --loadmodel './models/fadnet-ft-middlebury-in-0.001/model_best.pth' 

