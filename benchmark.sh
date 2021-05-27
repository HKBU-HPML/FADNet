net=slightfadnet

PY=python
CUDA_VISIBLE_DEVICES=0 $PY benchmark.py --devices 0 --net ${net} --trt
