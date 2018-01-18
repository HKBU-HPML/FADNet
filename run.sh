#python main.py --cuda --outf ./models-resnet --lr 0.01 --logFile train-resnet.log --showFreq 1 --devices 0,1,2,3
#python main.py --cuda --outf ./models-resnet --lr 0.01 --logFile train-resnet.log --showFreq 1 --devices 0,1,2,3 | tee resnet.log
# dispnet with resnet
# python main.py --cuda --outf ./models-resnet --lr 0.01 --logFile train-resnet.log --showFreq 1 --devices 0,1,2,3

# dispnetC with resnet
python main.py --cuda --outf ./models-dispC-resnet --lr 0.01 --logFile train-dispC-resnet.log --showFreq 1 --devices 0,1,2,3
