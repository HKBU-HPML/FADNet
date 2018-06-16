#python detecter_v3.py --model ./models/girl-domain-transfer-models-dispCSRes/model_best.pth --rp ./detect_result/domain_adaptation --filepath /home/datasets/imagenet/dispnet/virtual --filelist ./lists/girl09.list
#python detecter_v3.py --model ./models/girl-domain-transfer-models-dispCSRes/model_best.pth --rp ./detect_result/domain_adaptation --filepath /home/datasets/imagenet/dispnet/virtual --filelist ./lists/girl09_TEST.list

# TITANX
#python detecter_v3.py --model ./models/girl-domain-transfer-models-dispCSRes/model_best.pth --rp ./detect_result/domain_adaptation --filepath ~/dispflownet-release/data/girl --filelist ./lists/girl09.list
#python detecter_v3.py --model ./models/girl-domain-transfer-models-dispCSRes-finetune2/model_best.pth --rp ./detect_result/domain_adaptation-finetune2 --filepath ~/dispflownet-release/data/girl --filelist lists/real_release.list
python detecter_v3.py --model /data2/models/girl-domain-transfer-models-dispCSRes-correct-finetune1/model_best.pth --rp /data2/models/detect_result/domain_adaptation-correct-finetune1 --filepath ~/dispflownet-release/data/girl --filelist lists/girl09.list
