dnn="${dnn:-fadnet}"
nworkers="${nworkers:-4}"
source exp_configs/$dnn.conf
MPIPATH=/home/esetstore/.local/openmpi-4.0.1
PY=/home/esetstore/fadnet/bin/python
lr=6e-4

rdma="${rdma:-0}"
if [ "$rdma" = "0" ]; then
params="-mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_include 192.168.0.1/24 \
    -x NCCL_DEBUG=INFO  \
    -x NCCL_SOCKET_IFNAME=enp136s0f0,enp137s0f0 \
    -x NCCL_IB_DISABLE=1 \
    -x HOROVOD_CACHE_CAPACITY=0"
else
params="--mca pml ob1 --mca btl openib,vader,self --mca btl_openib_allow_ib 1 \
    -mca btl_tcp_if_include ib0 \
    --mca btl_openib_want_fork_support 1 \
    -x LD_LIBRARY_PATH  \
    -x NCCL_IB_DISABLE=0 \
    -x NCCL_SOCKET_IFNAME=ib0 \
    -x NCCL_DEBUG=INFO"
fi


$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile cluster${nworkers} -bind-to none -map-by slot \
    $params \
    $PY -W ignore dist_main.py --cuda --net $net --loss $loss --lr $lr \
               --outf $outf_model --logFile $logf \
               --devices $devices --batch_size $batchSize \
               --datapath $datapath \
               --dataset $dataset --trainlist $trainlist --vallist $vallist \
               --startRound $startR --startEpoch $startE \
               --model $model \
               --maxdisp $maxdisp \
	       --manualSeed 1024 \
	       --nwpernode 4 \

