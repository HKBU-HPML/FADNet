# test 1: dispnet with resnet
# python main.py --cuda --outf ./models-resnet --lr 0.01 --logFile train-resnet.log --showFreq 1 --devices 0,1,2,3

# test 2: dispnetC with resnet
# python main.py --cuda --outf ./models-dispC-resnet --lr 0.01 --logFile train-dispC-resnet.log --showFreq 1 --devices 0,1,2,3

# test 3: dispnetC with resnet, use model from test 2(23 epoches) to initialize
python main.py --cuda --outf ./models-dispC-resnet-finetune --lr 1e-7 --logFile train-dispC-resnet-finetune.log --showFreq 1 --devices 0,1,2,3 --model ./models-dispC-resnet-finetune/model_best.pth --startEpoch 61 --endEpoch 80
