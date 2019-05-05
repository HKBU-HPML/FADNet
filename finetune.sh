
#python kitti_finetune.py --maxdisp 192 \
#                   --model dispnetcres \
#                   --devices 0,1,2 \
#                   --datatype 2015 \
#                   --datapath data/KITTI_release/training/ \
#                   --epochs 300 \
#                   --loadmodel ./trained/finetune_273.tar \
#                   --savemodel ./trained/

python kitti_submission.py --maxdisp 192 \
                     --model dispnetcres \
                     --KITTI 2015 \
                     --datapath /data/KITTI_release/testing/ \
                     --loadmodel trained/best.tar \
                     --savepath submit_results/dispnetcres_KITTI2015/
