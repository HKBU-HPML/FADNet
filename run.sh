# test 1: dispnet with resnet
#python main.py --cuda --outf ./models-resnet --lr 0.01 --logFile train-resnet.log --showFreq 1 --devices 0,1,2,3
#python main.py --cuda --outf ./models-resnet --lr 0.01 --logFile train-resnet.log --showFreq 1 --devices 0,1,2,3 | tee resnet.log
# dispnet with resnet
# python main.py --cuda --outf ./models-resnet --lr 0.01 --logFile train-resnet.log --showFreq 1 --devices 0,1,2,3

# test 2: dispnetC with shrink resnet
#python main.py --cuda --outf ./models-dispC-resnet --lr 0.01 --logFile train-dispC-resnet.log --showFreq 1 --devices 0,1,2,3

# test 3: dispnetCSRes with shrink resnet
# python main.py --cuda --outf ./models-dispCSRes --lr 1e-4 --logFile train-dispCSRes.log --showFreq 1 --devices 0,1,2,3 --model ./models-dispCSRes/model_best.pth --startEpoch 30

#python main.py --cuda --outf ./models-dispCSRes --lr 0.0001 --logFile train-dispCSRes.log --showFreq 1 --devices 0,1,2,3

# test 4: dispnetC with shrink resnet + clean data
python main.py --cuda --outf ./cleandata-models-dispC-resnet-b64 --lr 0.0001 --logFile cleandata-train-dispC-resnet-b64.log --showFreq 1 --devices 0,1,2,3 --trainlist CLEAN_FlyingThings3D_release_TRAIN.list --vallist CLEAN_FlyingThings3D_release_TEST.list --batchSize 64

#python main.py --cuda --outf ./cleandata-models-dispCSRes --lr 0.0001 --logFile cleandata-train-dispCSRes.log --showFreq 1 --devices 0,1,2,3 --trainlist CLEAN_FlyingThings3D_release_TRAIN.list --vallist CLEAN_FlyingThings3D_release_TEST.list --model ./cleandata-models-dispCSRes/model_best.pth --startEpoch 5

# test 5: dispnetC with shrink resnet + dropout + clean data
# python main.py --cuda --outf ./cleandata-models-dispCSRes-dropout --lr 0.0001 --logFile cleandata-train-dispCSRes-dropout.log --showFreq 1 --devices 0,1,2,3 --trainlist CLEAN_FlyingThings3D_release_TRAIN.list --vallist CLEAN_FlyingThings3D_release_TEST.list
