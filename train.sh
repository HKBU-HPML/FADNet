dnn="${dnn:-fadnet}"
source exp_configs/$dnn.conf

python -W ignore main.py --cuda --net $net --loss $loss --lr $lr \
               --outf $outf_model --logFile $logf \
               --devices $devices --batch_size $batchSize \
               --datapath $datapath \
               --dataset $dataset --trainlist $trainlist --vallist $vallist \
               --startRound $startR --startEpoch $startE \
               --model $model \
               --maxdisp $maxdisp \
	       --manualSeed 1024 \

