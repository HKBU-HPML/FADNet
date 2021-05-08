CUDA_VISIBLE_DEVICES=0 python predict.py \
                --middlebury=1    --maxdisp=408 \
                --crop_height=1024  --crop_width=1536  \
                --datapath='./data/MiddEval3/' \
                --list='./lists/middeval3_train.list' \
                --savepath='./predict/middlebury/images/' \
                --loadmodel './models/fadnet-ft-middlebury/model_best.pth' 

