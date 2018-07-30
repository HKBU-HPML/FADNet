#python detecter_v3.py --model ./models/girl-domain-transfer-models-dispCSRes/model_best.pth --rp ./detect_result/domain_adaptation --filepath /home/datasets/imagenet/dispnet/virtual --filelist ./lists/girl09.list
#python detecter_v3.py --model ./models/girl-domain-transfer-models-dispCSRes/model_best.pth --rp ./detect_result/domain_adaptation --filepath /home/datasets/imagenet/dispnet/virtual --filelist ./lists/girl09_TEST.list

# TITANX
#python detecter_v3.py --model ./models/girl-domain-transfer-models-dispCSRes/model_best.pth --rp ./detect_result/domain_adaptation --filepath ~/dispflownet-release/data/girl --filelist ./lists/girl09.list
#python detecter_v3.py --model ./models/girl-domain-transfer-models-dispCSRes-finetune2/model_best.pth --rp ./detect_result/domain_adaptation-finetune2 --filepath ~/dispflownet-release/data/girl --filelist lists/real_release.list
python detecter_v3.py --model /home/comp/csshshi/fromsz/model_best.pth --rp ./detect_result/domain_adaptation --filepath /home/datasets/imagenet/dispnet --filelist lists/real_release.list
#python detecter_v3.py --model ./models/girl-domain-transfer-models-dispCSRes/model_best.pth --rp ./detect_result/domain_adaptation_1024 --filepath /home/datasets/imagenet/dispnet --filelist lists/real_release.list
