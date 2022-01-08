PY=/home/esetstore/fadnet/bin/python
CUDA_VISIBLE_DEVICES=1 $PY predict.py \
                --eth3d=1    --maxdisp=192 \
                --crop_height=576  --crop_width=960  \
                --datapath='./data/' \
                --list='./lists/eth3d_test.list' \
                --savepath='./predict/eth3d/' \
                --loadmodel './models/fadnet-ft-rvc-16e-4-mdq/model_best.pth' 
                #--loadmodel '/home/esetstore/repos/FADNet/models/fadnet-ft-rvc-4e-4-largemd/model_best.pth' 

