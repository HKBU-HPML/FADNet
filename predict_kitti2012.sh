PY=/home/esetstore/fadnet/bin/python
CUDA_VISIBLE_DEVICES=1 $PY predict.py \
                --kitti2012=1    --maxdisp=192 \
                --crop_height=384  --crop_width=1280  \
                --datapath='./data/' \
                --list='./lists/kitti2012_test.list' \
                --savepath='./predict/kitti2012/' \
                --loadmodel './models/fadnet-ft-rvc-16e-4-mdq/model_best.pth' 
                #--loadmodel '/home/esetstore/repos/FADNet/models/fadnet-ft-rvc-4e-4-largemd/model_best.pth' 

