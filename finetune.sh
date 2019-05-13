
python kitti_finetune.py --maxdisp 192 \
                   --model dispnetcres \
                   --devices 0,1,2,3 \
                   --datatype 2015 \
                   --datapath data/KITTI_release/training/ \
                   --epochs 0 \
                   --savemodel ./trained/dispnet-imagenet-argument/ \
                   --loss loss_configs/dispnetcres_flying.json \
                   #--loadmodel ./trained/dispnet-imagenet-argument/best.tar 
                   #--loadmodel ./models/dispCSRes/model_best.pth 

python kitti_submission.py --maxdisp 192 \
                     --model dispnetcres \
                     --KITTI 2015 \
                     --datapath /data/KITTI_release/testing/ \
                     --savepath submit_results/dispnetcres_imagenet_argument-pad_KITTI2015/ \
                     --loadmodel trained/dispnet-imagenet-argument/best.tar \
