PY=/home/esetstore/fadnet/bin/python
CUDA_VISIBLE_DEVICES=1 $PY predict.py \
                --sceneflow=1    --maxdisp=192 \
                --crop_height=576  --crop_width=960  \
                --datapath='./data/' \
                --list='./lists/FlyingThings3D_release_TEST.list' \
                --savepath='./predict/sf/' \
                --loadmodel './models/fadnet-sceneflow.pth' 
                #--loadmodel '/home/esetstore/repos/FADNet/models/fadnet-ft-rvc-4e-4-largemd/model_best.pth' 

