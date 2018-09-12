# CMake generated Testfile for 
# Source directory: /home/comp/qiangwang/pytorch-dispnet/tools/lmbspecialops/test
# Build directory: /home/comp/qiangwang/pytorch-dispnet/tools/lmbspecialops/build/test
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_FloOp "/home/comp/qiangwang/anaconda3/bin/python" "test_FloOp.py")
set_tests_properties(test_FloOp PROPERTIES  ENVIRONMENT "LMBSPECIALOPS_LIB=/home/comp/qiangwang/pytorch-dispnet/tools/lmbspecialops/build/lib/lmbspecialops.so" WORKING_DIRECTORY "/home/comp/qiangwang/pytorch-dispnet/tools/lmbspecialops/test")
add_test(test_FlowToDepth2 "/home/comp/qiangwang/anaconda3/bin/python" "test_FlowToDepth2.py")
set_tests_properties(test_FlowToDepth2 PROPERTIES  ENVIRONMENT "LMBSPECIALOPS_LIB=/home/comp/qiangwang/pytorch-dispnet/tools/lmbspecialops/build/lib/lmbspecialops.so" WORKING_DIRECTORY "/home/comp/qiangwang/pytorch-dispnet/tools/lmbspecialops/test")
add_test(test_LeakyRelu "/home/comp/qiangwang/anaconda3/bin/python" "test_LeakyRelu.py")
set_tests_properties(test_LeakyRelu PROPERTIES  ENVIRONMENT "LMBSPECIALOPS_LIB=/home/comp/qiangwang/pytorch-dispnet/tools/lmbspecialops/build/lib/lmbspecialops.so" WORKING_DIRECTORY "/home/comp/qiangwang/pytorch-dispnet/tools/lmbspecialops/test")
add_test(test_Lz4 "/home/comp/qiangwang/anaconda3/bin/python" "test_Lz4.py")
set_tests_properties(test_Lz4 PROPERTIES  ENVIRONMENT "LMBSPECIALOPS_LIB=/home/comp/qiangwang/pytorch-dispnet/tools/lmbspecialops/build/lib/lmbspecialops.so" WORKING_DIRECTORY "/home/comp/qiangwang/pytorch-dispnet/tools/lmbspecialops/test")
add_test(test_Lz4Raw "/home/comp/qiangwang/anaconda3/bin/python" "test_Lz4Raw.py")
set_tests_properties(test_Lz4Raw PROPERTIES  ENVIRONMENT "LMBSPECIALOPS_LIB=/home/comp/qiangwang/pytorch-dispnet/tools/lmbspecialops/build/lib/lmbspecialops.so" WORKING_DIRECTORY "/home/comp/qiangwang/pytorch-dispnet/tools/lmbspecialops/test")
add_test(test_Median3x3Downsample "/home/comp/qiangwang/anaconda3/bin/python" "test_Median3x3Downsample.py")
set_tests_properties(test_Median3x3Downsample PROPERTIES  ENVIRONMENT "LMBSPECIALOPS_LIB=/home/comp/qiangwang/pytorch-dispnet/tools/lmbspecialops/build/lib/lmbspecialops.so" WORKING_DIRECTORY "/home/comp/qiangwang/pytorch-dispnet/tools/lmbspecialops/test")
add_test(test_PfmOp "/home/comp/qiangwang/anaconda3/bin/python" "test_PfmOp.py")
set_tests_properties(test_PfmOp PROPERTIES  ENVIRONMENT "LMBSPECIALOPS_LIB=/home/comp/qiangwang/pytorch-dispnet/tools/lmbspecialops/build/lib/lmbspecialops.so" WORKING_DIRECTORY "/home/comp/qiangwang/pytorch-dispnet/tools/lmbspecialops/test")
add_test(test_PpmOp "/home/comp/qiangwang/anaconda3/bin/python" "test_PpmOp.py")
set_tests_properties(test_PpmOp PROPERTIES  ENVIRONMENT "LMBSPECIALOPS_LIB=/home/comp/qiangwang/pytorch-dispnet/tools/lmbspecialops/build/lib/lmbspecialops.so" WORKING_DIRECTORY "/home/comp/qiangwang/pytorch-dispnet/tools/lmbspecialops/test")
add_test(test_ReplaceNonfinite "/home/comp/qiangwang/anaconda3/bin/python" "test_ReplaceNonfinite.py")
set_tests_properties(test_ReplaceNonfinite PROPERTIES  ENVIRONMENT "LMBSPECIALOPS_LIB=/home/comp/qiangwang/pytorch-dispnet/tools/lmbspecialops/build/lib/lmbspecialops.so" WORKING_DIRECTORY "/home/comp/qiangwang/pytorch-dispnet/tools/lmbspecialops/test")
add_test(test_ScaleInvariantGradient "/home/comp/qiangwang/anaconda3/bin/python" "test_ScaleInvariantGradient.py")
set_tests_properties(test_ScaleInvariantGradient PROPERTIES  ENVIRONMENT "LMBSPECIALOPS_LIB=/home/comp/qiangwang/pytorch-dispnet/tools/lmbspecialops/build/lib/lmbspecialops.so" WORKING_DIRECTORY "/home/comp/qiangwang/pytorch-dispnet/tools/lmbspecialops/test")
add_test(test_WebpOp "/home/comp/qiangwang/anaconda3/bin/python" "test_WebpOp.py")
set_tests_properties(test_WebpOp PROPERTIES  ENVIRONMENT "LMBSPECIALOPS_LIB=/home/comp/qiangwang/pytorch-dispnet/tools/lmbspecialops/build/lib/lmbspecialops.so" WORKING_DIRECTORY "/home/comp/qiangwang/pytorch-dispnet/tools/lmbspecialops/test")
