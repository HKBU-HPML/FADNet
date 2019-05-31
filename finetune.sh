
python kitti_finetune.py --maxdisp 192 \
                   --model dispnetcres \
                   --devices 0,1,2,3 \
                   --datatype 2015 \
                   --datapath /datasets/kitti/training/ \
                   --epochs 2000 \
                   --savemodel /datasets/sh_kittis/ShKittiTrained/dispnet-imagenet-argument2/ \
                   --loss loss_configs/kitti.json \
                   --loadmodel  ./models/ShDispCSRes/model_best.pth
                   #--loadmodel /datasets/sh_kittis/ShKittiTrained/dispnet-imagenet-argument/best.tar 

