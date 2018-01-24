# test 1: dispnet with resnet
#python main.py --cuda --outf ./models-resnet --lr 0.01 --logFile train-resnet.log --showFreq 1 --devices 0,1,2,3
#python main.py --cuda --outf ./models-resnet --lr 0.01 --logFile train-resnet.log --showFreq 1 --devices 0,1,2,3 | tee resnet.log
# dispnet with resnet
# python main.py --cuda --outf ./models-resnet --lr 0.01 --logFile train-resnet.log --showFreq 1 --devices 0,1,2,3

# test 2: dispnetC with shrink resnet
# python main.py --cuda --outf ./models-dispC-resnet --lr 0.01 --logFile train-dispC-resnet.log --showFreq 1 --devices 0,1,2,3

# test 3: dispnetCSRes with shrink resnet
python main.py --cuda --outf ./models-dispCSRes --lr 1e-4 --logFile train-dispCSRes-test.log --showFreq 1 --devices 0,1,2,3 --model ./models-dispCSRes/model_best.pth --startEpoch 30

