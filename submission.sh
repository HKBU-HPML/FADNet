#model_path=./trained/dispnetcres-imn-bottom-2015/best.tar
#save_path=./submit_results/dispnetcres_imn_bottom_0.597_KITTI2015_20190717/
model_path=./trained/dispCSRes-imn-KITTI2015-valavg/best.tar
save_path=./submit_results/dispnetcres_imn_avg_bottom_0.602_KITTI2015_20190816/
python kitti_submission.py --maxdisp 192 \
                     --model dispnetcres \
                     --KITTI 2015 \
                     --datapath /datasets/kitti/testing/ \
                     --savepath $save_path \
                     --loadmodel $model_path \
