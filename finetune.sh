
python kitti_finetune.py --maxdisp 192 \
                   --model dispnetcres \
                   --devices 0,1,2,3 \
                   --datatype 2015 \
                   --datapath data/kitti/training/ \
                   --epochs 0 \
                   --loss loss_configs/dispnetcres_kitti.json \
                   --savemodel ./trained/dispCSRes-snd/ \
                   --loadmodel ./trained/dispCSRes-snd/best.tar \
                   #--loadmodel ./models/dispCSRes/model_best.pth \
                   #--loadmodel ./trained/dispnetc-regression-snd/best.tar \

#python kitti_submission.py --maxdisp 192 \
#                     --model dispnetcres \
#                     --KITTI 2015 \
#                     --datapath data/kitti/testing/ \
#                     --savepath submit_results/dispCSRes-snd/ \
#                     --loadmodel trained/dispCSRes-snd/best.tar \
