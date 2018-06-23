python main.py --cuda --outf ./models/girl-ir-dispCSRes --lr 0.01 --logFile girl-ir-train-dispCSRes.log --showFreq 1 --devices 0,1,2,3 --trainlist ./lists/girl_with_ir2_TRAIN.list  --vallist ./lists/girl_with_ir2_TEST.list --startEpoch 0 --datapath /data2/virtual3 --batchSize 8 --endEpoch 200 --domain_transfer 0

