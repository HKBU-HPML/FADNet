## kitti 2015
python kitti_finetune.py --maxdisp 192 \
                   --model fadnet \
                   --devices 0,1 \
                   --datatype 2015 \
                   --datapath ./data/kitti2015/training/ \
                   --epochs 600 \
                   --loss loss_configs/fadnet_kitti.json \
                   --savemodel ./trained/fadnet-imn-KITTI2015-bottom/ \
                   --loadmodel ./models/fadnet-sf.pth \
                   #--loadmodel  ./trained/fadnet-imn-KITTI2015/finetune_2_501.tar \
                   #--loadmodel ./models/pretrained_sceneflow.tar \
                   #--datapath /datasets/kitti2015/training/ \
                   #--epochs 1200 \
                   #--loss loss_configs/fadnet_kitti.json \
                   #--loadmodel ./models/gwcnet-ft3d-irs.pth \
                   #--savemodel ./trained/fadnet-imn-KITTI2015-split/ \

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

#                   --model gwcnet \
#                   --devices 0,1 \
#                   --datatype 2012 \
#                   --datapath /datasets/kitti2012/training/ \
#                   --epochs 1200 \
#                   --savemodel trained/dispnetcres-imn-bottom-2012/ \
#                   --loss loss_configs/fadnet_kitti.json \
#                   --loadmodel  ./models/gwcnet-ft3d-irs.pth \

