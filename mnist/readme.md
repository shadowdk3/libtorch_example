# MNIST

This folder contains an example of training a computer vision model to recognize
digits in images from the MNIST dataset, using the PyTorch C++ frontend.

- train model
- test model
- save model

## Environment 

- Jetson ORIN AGX
- Pytorch

## Install dependencies

- torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
- torchvision-0.15.1-cp38-cp38-manylinux2014_aarch64.whl

## Run
```
cd mnist
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
cmake --build . --config Release
```

## Result

```
Train Epoch: 1 [59584/60000] Loss: 0.2271
Test set: Average loss: 0.2042 | Accuracy: 0.936
Train Epoch: 2 [59584/60000] Loss: 0.1991
Test set: Average loss: 0.1305 | Accuracy: 0.959
Train Epoch: 3 [59584/60000] Loss: 0.1568
Test set: Average loss: 0.1023 | Accuracy: 0.969
Train Epoch: 4 [59584/60000] Loss: 0.0724
Test set: Average loss: 0.0908 | Accuracy: 0.971
Train Epoch: 5 [59584/60000] Loss: 0.1081
Test set: Average loss: 0.0796 | Accuracy: 0.973
Train Epoch: 6 [59584/60000] Loss: 0.0901
Test set: Average loss: 0.0723 | Accuracy: 0.975
Train Epoch: 7 [59584/60000] Loss: 0.0463
Test set: Average loss: 0.0668 | Accuracy: 0.979
Train Epoch: 8 [59584/60000] Loss: 0.0968
Test set: Average loss: 0.0593 | Accuracy: 0.981
Train Epoch: 9 [59584/60000] Loss: 0.0513
Test set: Average loss: 0.0582 | Accuracy: 0.981
Train Epoch: 10 [59584/60000] Loss: 0.0450
Test set: Average loss: 0.0546 | Accuracy: 0.983
```