#net=dispnetc
#net=crl
#
net=psmnet
#net=ganet
#net=gwcnet
#
#net=fadnet
#net=mobilefadnet
#net=slightfadnet
#net=tinyfadnet

#PY=/home/esetstore/fadnet/bin/python
PY=python
CUDA_VISIBLE_DEVICES=0 $PY benchmark.py --devices 0 --net ${net} #--trt
