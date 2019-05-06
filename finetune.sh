
#python kitti_finetune.py --maxdisp 192 \
#                   --model dispnetcres \
#                   --devices 0,1,2 \
#                   --datatype 2015 \
#                   --datapath data/KITTI_release/training/ \
#                   --epochs 900 \
#                   --savemodel ./trained/ \
#                   --loadmodel ./models/model_best.pth \

python kitti_submission.py --maxdisp 192 \
                     --model dispnetcres \
                     --KITTI 2015 \
                     --datapath /data/KITTI_release/testing/ \
                     --savepath submit_results/dispnetcres_KITTI2015/ \
                     --loadmodel trained/best.tar \
