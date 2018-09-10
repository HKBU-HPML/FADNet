# lmbspecialops

[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)

lmbspecialops is a collection of tensorflow ops.
The ops focus on networks for predicting depth and camera motion as in DeMoN, but many can also be useful for other tasks.

If you use this code for research please cite:
   
    @InProceedings{UZUMIDB17,
      author       = "B. Ummenhofer and H. Zhou and J. Uhrig and N. Mayer and E. Ilg and A. Dosovitskiy and T. Brox",
      title        = "DeMoN: Depth and Motion Network for Learning Monocular Stereo",
      booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
      month        = " ",
      year         = "2017",
      url          = "http://lmb.informatik.uni-freiburg.de//Publications/2017/UZUMIDB17"
    }


See the [Op documentation](doc/lmbspecialops_doc.md) for a description of all functions.



## Requirements

Building and using lmbspecialops depends on the following libraries and programs

    tensorflow 1.4.0
    cmake 3.7.1
    python 3.5
    cuda 8.0.61 (required for building with gpu support)

The versions match the configuration we have tested on an ubuntu 16.04 system.
lmbspecialops can work with other versions of the aforementioned dependencies, e.g. tensorflow 1.3, but this is not well tested.



## Build instructions


Checkout the repository and setup the build directory.

```bash
git clone https://github.com/lmb-freiburg/lmbspecialops.git
cd lmbspecialops

mkdir build
cd build
cmake ..
# cmake .. -DBUILD_WITH_CUDA=OFF # to disable gpu support
```

Use ```ccmake``` to customize the configuration.
You may want to adapt GPU code generation to your graphics card.
E.g. to enable code generation for the Tesla K80 enable *GENERATE_KEPLER_SM37_CODE*

```bash
ccmake .
```

Build the library

```bash
make
```

To use the ops, you need to add the ```lmbspecialops/python``` directory to your python path.
Then you can import and use the ops like this 

```python
import lmbspecialops
import tensorflow as tf
import numpy as np

tf.InteractiveSession()

A = tf.constant([1,2,np.nan])
B = lmbspecialops.replace_nonfinite(A)
print(B.eval()) # prints [1, 2, 0]
```

    

### Virtualenv build 

To build inside a virtualenv make sure to activate the environment before
running cmake.

If you encounter problems with cmake not finding the correct paths to your
python interpreter or tensorflow see https://gist.github.com/datagrok/2199506
for possible solutions.



## License

lmbspecialops is under the [GNU General Public License v3.0](LICENSE.txt)

