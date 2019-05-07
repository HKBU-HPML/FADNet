
python kitti_finetune.py --maxdisp 192 \
                   --model dispnetcres \
                   --devices 0,1,2,3 \
                   --datatype 2015 \
                   --datapath data/KITTI_release/training/ \
                   --epochs 900 \
                   --savemodel ./trained/dispnet-imagenet/ \
                   --loadmodel ./models/dispnet-imagenet-best.pth \
                   --loss loss_configs/dispnetcres_flying.json \

#python kitti_submission.py --maxdisp 192 \
#                     --model dispnetcres \
#                     --KITTI 2015 \
#                     --datapath /data/KITTI_release/testing/ \
#                     --savepath submit_results/dispnetcres_imagenet_finetune300_KITTI2015/ \
#                     --loadmodel trained/dispnet-imagenet/best.tar \
