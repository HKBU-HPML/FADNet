#!/bin/bash
#cd ./correlation_package
#python setup.py install --user
cd ./resample2d_package 
sudo python3 setup.py install 
cd ../channelnorm_package 
sudo python3 setup.py install 
cd ..
