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

#test 4: dispnetC with shrink resnet + clean data
#python main.py --cuda --outf ./no-occ-models-dispCSRes-finetune --lr 0.00001 --logFile no-occ-train-dispCSRes-finetune.log --showFreq 1 --devices 0,1,2,3 --trainlist ./lists/SHAOHUAI_CLEAN_FlyingThings3D_release_TRAIN.list --vallist ./lists/SHAOHUAI_CLEAN_FlyingThings3D_release_TEST.list --model ./cleandata-dispCSRes-model_best.pth --startEpoch 1

#test 4-1: dispnetC with shrink resnet + clean data, finetune girl data
#python main.py --cuda --outf ./girl-models-dispCSRes-finetune --lr 0.001 --logFile girl-train-dispCSRes-finetune.log --showFreq 1 --devices 0,1,2,3 --trainlist ./lists/girl.list --vallist ./lists/girl_TEST.list --model ./cleandata-dispCSRes-model_best.pth --startEpoch 1 --datapath /home/datasets/imagenet/dispnet/virtual/girl --batchSize 8
#python main.py --cuda --outf ./girl-models-dispCSRes-finetune-changeweight-2nd --lr 0.001 --logFile girl-train-dispCSRes-finetune-changeweight-2nd.log --showFreq 1 --devices 0,1,2,3 --trainlist ./lists/girl.list --vallist ./lists/girl_TEST.list --model ./girl-models-dispCSRes-finetune-changeweight/model_best.pth --startEpoch 1 --datapath /home/datasets/imagenet/dispnet/virtual/girl --batchSize 8
#python main.py --cuda --outf ./girl-models-dispCSRes-finetune-changeweight-2nd --lr 4.8828125e-07 --logFile girl-train-dispCSRes-finetune-changeweight-2nd.log --showFreq 1 --devices 0,1,2,3 --trainlist ./lists/girl.list --vallist ./lists/girl_TEST.list --model ./girl-models-dispCSRes-finetune-changeweight/model_best.pth --startEpoch 40 --datapath /home/datasets/imagenet/dispnet/virtual/girl --batchSize 8 --endEpoch 100
#python main.py --cuda --outf ./models/girl02-models-dispCSRes-finetune-changeweight --lr 0.001 --logFile girl02-train-dispCSRes-finetune-changeweight.log --showFreq 1 --devices 0,1,2,3 --trainlist ./lists/girl02.list --vallist ./lists/girl02_TEST.list --model ./models/girl-models-dispCSRes-finetune-changeweight/model_best.pth --startEpoch 1 --datapath /home/datasets/imagenet/dispnet/virtual --batchSize 8 --endEpoch 100
#python main.py --cuda --outf ./models/girl02-models-dispCSRes-finetune-changeweight-2nd --lr 0.0005 --logFile girl02-train-dispCSRes-finetune-changeweight-2nd.log --showFreq 1 --devices 0,1,2,3 --trainlist ./lists/girl02.list --vallist ./lists/girl02_TEST.list --model ./models/girl02-models-dispCSRes-finetune-changeweight/model_best.pth --startEpoch 1 --datapath /home/datasets/imagenet/dispnet/virtual --batchSize 8 --endEpoch 100
python main.py --cuda --outf ./models/girl02-models-dispCSRes-finetune-changeweight-3nd --lr 0.0005 --logFile girl02-train-dispCSRes-finetune-changeweight-3nd.log --showFreq 1 --devices 0,1,2,3 --trainlist ./lists/girl02.list --vallist ./lists/girl02_TEST.list --model ./models/girl02-models-dispCSRes-finetune-changeweight-2nd/model_best.pth --startEpoch 1 --datapath /home/datasets/imagenet/dispnet/virtual --batchSize 8 --endEpoch 100

# test 5: dispnetC with shrink resnet + dropout + clean data
#python main.py --cuda --outf ./cleandata-models-dispCSRes-dropout --lr 0.0001 --logFile cleandata-train-dispCSRes-dropout.log --showFreq 1 --devices 0,1 --trainlist CLEAN_FlyingThings3D_release_TRAIN.list --vallist CLEAN_FlyingThings3D_release_TEST.list
#python main.py --cuda --outf ./cleandata-models-dispC-resnet-b64 --lr 0.0001 --logFile cleandata-train-dispC-resnet-b64.log --showFreq 1 --devices 0,1,2,3 --trainlist CLEAN_FlyingThings3D_release_TRAIN.list --vallist CLEAN_FlyingThings3D_release_TEST.list --batchSize 64

#python main.py --cuda --outf ./cleandata-models-dispCSRes --lr 0.0001 --logFile cleandata-train-dispCSRes.log --showFreq 1 --devices 0,1,2,3 --trainlist CLEAN_FlyingThings3D_release_TRAIN.list --vallist CLEAN_FlyingThings3D_release_TEST.list --model ./cleandata-models-dispCSRes/model_best.pth --startEpoch 5

# test 5: dispnetC with shrink resnet + dropout + clean data
# python main.py --cuda --outf ./cleandata-models-dispCSRes-dropout --lr 0.0001 --logFile cleandata-train-dispCSRes-dropout.log --showFreq 1 --devices 0,1,2,3 --trainlist CLEAN_FlyingThings3D_release_TRAIN.list --vallist CLEAN_FlyingThings3D_release_TEST.list

# test 5: dispnetC with shrink resnet + clean data + occulution
# python main.py --cuda --outf ./cc-models-dispC-resnet-clean --lr 0.0001 --logFile cc-train-dispC-resnet-clean.log --showFreq 1 --devices 0,1,2,3 --trainlist CC_FlyingThings3D_release_TRAIN.list --vallist CC_FlyingThings3D_release_TEST.list 
# python main.py --cuda --outf ./cc-models-dispC-resnet --lr 0.0002 --logFile cc-train-dispC-resnet-sr.log --showFreq 1 --devices 0,1,2,3 --trainlist CC_FlyingThings3D_release_TRAIN.list --vallist CC_FlyingThings3D_release_TEST.list --model ./cc-models-dispC-resnet/model_best.pth --startEpoch 50 --endEpoch 80

# test 6: dispnetC with shrink resnet + clean data + occulution + finetune on cleandata model
# python main.py --cuda --outf ./cc-models-dispC-resnet-cleandata-finetune --lr 1e-5 --logFile cc-train-dispC-resnet-cleandata-finetune.log --showFreq 1 --devices 0,1,2,3 --trainlist CC_FlyingThings3D_release_TRAIN.list --vallist CC_FlyingThings3D_release_TEST.list --model ./cleandata-models-dispCSRes-exp/model_best.pth

# test 7: dispnetC with shrink resnet + sgm mix finetune on cleandata model
#python main.py --cuda --outf ./cc-models-dispC-resnet-cleandata-mix-cont --lr 1e-5 --logFile cc-train-dispC-resnet-cleandata-mix-cont.log --showFreq 1 --devices 0,1,2,3 --trainlist mix_sgm_release_TRAIN.list --vallist mix_sgm_release_TEST.list --model ./cc-models-dispC-resnet-cleandata-mix-cont/model_best.pth

# Relu
#python main.py --cuda --outf /data/cc-models-dispC-resnet-relu --lr 0.0001 --logFile cc-train-dispC-resnet-relu-cont.log --showFreq 1 --devices 0,1 --trainlist CC_FlyingThings3D_release_TRAIN.list --vallist CC_FlyingThings3D_release_TEST.list --model  /data/cc-models-dispC-resnet-relu/model_best.pth --startEpoch 12

# Relu + finetune
#python main.py --cuda --outf /data/cc-models-dispC-resnet-relu-ft --lr 0.0001 --logFile cc-train-dispC-resnet-relu-cont-ft.log --showFreq 1 --devices 0,1 --trainlist CC_FlyingThings3D_release_TRAIN.list --vallist CC_FlyingThings3D_release_TEST.list --model  /data/cc-models-dispC-resnet-relu/model_best.pth --startEpoch 50 --endEpoch 100

# cleandata-1.68-model + Relu + finetune + remove black
# python main.py --cuda --outf ./cleandata-models-dispC-resnet-relu-ft-rb --lr 0.0001 --logFile cleandata-train-dispC-resnet-relu-cont-ft-rb.log --showFreq 1 --devices 0,1,2,3 --trainlist RB_FlyingThings3D_release_TRAIN.list --vallist RB_FlyingThings3D_release_TEST.list --model ./cleandata-models-dispC-resnet-relu-ft-rb/model_best.pth --startEpoch 35 --endEpoch 100

# cleandata-1.68-model + Relu + finetune + remove black + real sgm
# python main.py --cuda --outf ./cleandata-models-dispC-resnet-relu-ft-rb-mix --lr 1e-5 --logFile cleandata-train-dispC-resnet-relu-cont-ft-rb-mix.log --showFreq 1 --devices 0,1,2,3 --trainlist mix_sgm_release_TRAIN.list --vallist mix_sgm_release_TEST.list --model ./cleandata-models-dispC-resnet-relu-ft-rb-mix/model_best.pth --startEpoch 0 --endEpoch 100


# cleandata-1.68-model + Relu + girl finetune
#python main.py --cuda --outf ./cleandata-models-dispCSR-girl --lr 1e-5 --logFile cleandata-train-dispCSR-girl.log --showFreq 1 --devices 0,1,2,3 --trainlist girl_TRAIN.list --vallist girl_TEST.list --model ./cleandata-models-dispCSR-exp/model_best.pth --startEpoch 0 --endEpoch 100
#python main.py --cuda --outf ./models/cleandata-models-dispCSR-girl --lr 1e-4 --logFile cleandata-train-dispCSR-girl-cont.log --showFreq 1 --devices 0,1,2,3 --trainlist lists/girl_TRAIN.list --vallist lists/girl_TEST.list --model ./models/cleandata-models-dispCSR-girl/model_best.pth --startEpoch 0 --endEpoch 60
