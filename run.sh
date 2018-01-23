# test 1: dispnet with resnet
#python main.py --cuda --outf ./models-resnet --lr 0.01 --logFile train-resnet.log --showFreq 1 --devices 0,1,2,3
#python main.py --cuda --outf ./models-resnet --lr 0.01 --logFile train-resnet.log --showFreq 1 --devices 0,1,2,3 | tee resnet.log
# dispnet with resnet
# python main.py --cuda --outf ./models-resnet --lr 0.01 --logFile train-resnet.log --showFreq 1 --devices 0,1,2,3

# test 2: dispnetC with shrink resnet
#python main.py --cuda --outf ./models-dispC-resnet --lr 0.01 --logFile train-dispC-resnet.log --showFreq 1 --devices 0,1,2,3

# test 3: dispnetCSRes with shrink resnet
#python main.py --cuda --outf ./models-dispCSRes --lr 0.0001 --logFile train-dispCSRes.log --showFreq 1 --devices 0,1,2,3

# test 4: dispnetC with shrink resnet + clean data
python main.py --cuda --outf ./cleandata-models-dispC-resnet --lr 0.0001 --logFile cleandata-train-dispC-resnet.log --showFreq 1 --devices 0,1,2,3 --trainlist CLEAN_FlyingThings3D_release_TRAIN.list --vallist CLEAN_FlyingThings3D_release_TEST.list
