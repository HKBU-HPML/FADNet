net=mobilefadnet

PY=/home/esetstore/fadnet/bin/python
CUDA_VISIBLE_DEVICES=0 $PY benchmark.py --devices 0 --net ${net} 
