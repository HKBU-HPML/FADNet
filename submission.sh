#model_path=/home/esetstore/repos/fadnet-shyhuai/models/fadnet-sceneflow-n32sf/model_best.pth
#save_path=./submit_results/fadnet-ft-kitti-KITTI2015/
model_path=./models/fadnet-kitti2015/model_best.pth
save_path=./submit_results/fadnet-dist/
net=fadnet
#model_path=./trained/psmnet-imn-KITTI2015-split/best.tar
#save_path=./submit_results/psmnet-imn-KITTI2015-split/
#net=psmnet
python kitti_submission.py --maxdisp 192 \
                     --model $net \
                     --KITTI 2015 \
                     --datapath /datasets/kitti2015/testing/ \
                     --savepath $save_path \
                     --loadmodel $model_path \
