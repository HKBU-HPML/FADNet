model_path=./trained/fadnet-imn-KITTI2015-bottom/best.tar
save_path=./submit_results/fadnet-imn-KITTI2015-bottom/
net=fadnet
#model_path=./trained/psmnet-imn-KITTI2015-split/best.tar
#save_path=./submit_results/psmnet-imn-KITTI2015-split/
#net=psmnet
python kitti_submission.py --maxdisp 192 \
                     --model $net \
                     --KITTI 2015 \
                     --datapath ./data/kitti2015/testing/ \
                     --savepath $save_path \
                     --loadmodel $model_path \
