## kitti 2015
python kitti_finetune.py --maxdisp 192 \
                   --model fadnet \
                   --devices 0,1,2,3 \
                   --datatype 2015 \
                   --datapath ./data/kitti2015/training/ \
                   --epochs 600 \
                   --loss loss_configs/fadnet_kitti.json \
                   --savemodel ./trained/fadnet-imn-KITTI2015-bottom/ \
                   --loadmodel ./models/fadnet-sf.pth \
                   #--loadmodel  ./trained/fadnet-imn-KITTI2015/finetune_2_501.tar \
                   #--loadmodel ./models/pretrained_sceneflow.tar \
                   #--loadmodel ./trained/fadnet-imn-KITTI2015-split/best.tar \
                   #--loadmodel ./models/dispCSRes-imn/model_best.pth \
                   #--loadmodel  ./models/ShDispCSRes/model_best.pth
                   #--loadmodel /datasets/sh_kittis/ShKittiTrained/dispnet-imagenet-argument/best.tar 

# kitti 2012
#python kitti_finetune.py --maxdisp 192 \
#                   --model fadnet \
#                   --devices 0,1,2,3 \
#                   --datatype 2012 \
#                   --datapath ./data/kitti2012/training/ \
#                   --epochs 600 \
#                   --loss loss_configs/fadnet_kitti.json \
#                   --savemodel trained/fadnet-imn-KITTI2012-bottom/ \
#                   --loadmodel ./models/fadnet-sf.pth \
#                   --loadmodel  ./trained/fadnet-imn-KITTI2012/finetune_2_401.tar \

