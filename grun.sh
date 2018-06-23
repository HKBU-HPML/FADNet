# From zero to train girl data, with data augment
python main.py --cuda --outf ./models/girl-models-dispCSRes-crop-1nd --lr 0.0001 --logFile girl-train-dispCSRes-crop-1nd.log --showFreq 1 --devices 0,1,2,3 --trainlist ./lists/girl20_TRAIN.list --vallist ./lists/girl20_TEST.list --startEpoch 0 --datapath /home/datasets/imagenet/dispnet/virtual --batchSize 8 --endEpoch 200 --augment 1
