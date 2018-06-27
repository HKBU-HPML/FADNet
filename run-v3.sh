#python main.py --cuda --outf ./models/girl-domain-transfer-models-dispCSRes --lr 0.0001 --logFile girl-domain-transfer-train-dispCSRes.log --showFreq 1 --devices 0,1,2,3 --trainlist ./lists/girl20_TRAIN.list --vallist ./lists/girl20_TEST.list --startEpoch 0 --datapath /home/datasets/imagenet/dispnet/virtual --batchSize 8 --endEpoch 200 --domain_transfer 1 --tdlist lists/real_release.list

# TITANX
#python main.py --cuda --outf ./models/girl-domain-transfer-models-dispCSRes-correct --lr 0.0001 --logFile girl-domain-transfer-train-dispCSRes-correct.log --showFreq 1 --devices 0,1,2,3 --trainlist ./lists/girl09.list --vallist ./lists/girl09_TEST.list --startEpoch 0 --datapath ~/dispflownet-release/data/girl --batchSize 8 --endEpoch 200 --domain_transfer 1 --tdlist lists/real_release.list

# Finetune with weight: loss_weights = (0.9, 0.05, 0.02, 0.02, 0.01, 0.005, 0.0025)
#python main.py --cuda --outf /data2/models/girl-domain-transfer-models-dispCSRes-correct-finetune1 --lr 0.0001 --logFile girl-domain-transfer-train-dispCSRes-correct-finetune1.log --model /data2/models/girl-domain-transfer-models-dispCSRes-correct/model_best.pth --showFreq 1 --devices 0,1,2,3 --trainlist ./lists/girl09.list --vallist ./lists/girl09_TEST.list --startEpoch 0 --datapath ~/dispflownet-release/data/girl --batchSize 8 --endEpoch 200 --domain_transfer 1 --tdlist lists/real_release.list

# Finetune with weight: loss_weights = (0.32, 0.16, 0.08, 0.04, 0.02, 0.01, 0.005)
#python main.py --cuda --outf ./models/girl-domain-transfer-models-dispCSRes-finetune1 --lr 0.0001 --logFile girl-domain-transfer-train-dispCSRes-finetune1.log --model ./models/girl-domain-transfer-models-dispCSRes/model_best.pth --showFreq 1 --devices 0,1,2,3 --trainlist ./lists/girl09.list --vallist ./lists/girl09_TEST.list --startEpoch 0 --datapath ~/dispflownet-release/data/girl --batchSize 8 --endEpoch 200 --domain_transfer 1 --tdlist lists/real_release.list

# Finetune with weight: loss_weights = (0.8, 0.1, 0.04, 0.04, 0.02, 0.01, 0.005)
#python main.py --cuda --outf ./models/girl-domain-transfer-models-dispCSRes-finetune2 --lr 0.0001 --logFile girl-domain-transfer-train-dispCSRes-finetune2.log --model ./models/girl-domain-transfer-models-dispCSRes-finetune1/model_best.pth --showFreq 1 --devices 0,1,2,3 --trainlist ./lists/girl09.list --vallist ./lists/girl09_TEST.list --startEpoch 0 --datapath ~/dispflownet-release/data/girl --batchSize 8 --endEpoch 200 --domain_transfer 1 --tdlist lists/real_release.list

