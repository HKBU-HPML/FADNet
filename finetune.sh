
python kitti_finetune.py --maxdisp 192 \
                   --model dispnetcres \
                   --devices 0,1,2,3 \
                   --datatype 2015 \
                   --datapath /datasets/kitti/training/ \
                   --epochs 2000 \
                   --savemodel /datasets/sh_kittis/ShKittiTrained/dispnet-imagenet-argument/ \
                   --loss loss_configs/kitti.json \
                   --loadmodel  /datasets/sh_kittis/models/ShDispCSRes/model_best.pth

#python kitti_submission.py --maxdisp 192 \
#                     --model dispnetcres \
#                     --KITTI 2015 \
#                     --datapath /datasets/kitti/testing/ \
#                     --savepath /datasets/sh_kittis/submit_results/dispnetcres_imagenet_argument-pad_KITTI2015/ \
#                     --loadmodel /datasets/sh_kittis/ShKittiTrained/dispnet-imagenet-argument/best.tar \
