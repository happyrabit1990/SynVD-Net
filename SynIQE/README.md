# SynIQE-PyTorch

This is a PyTorch implementation of the TIP2018 paper [*Convolutional neural network-based synthesized view quality enhancement for 3D video coding*](https://ieeexplore.ieee.org/document/8416728). 


## How to run

### 1. Dependences
* [PyTorch](http://pytorch.org/)
* [torchvision](https://github.com/pytorch/vision)
* OpenCV for Python
* [HDF5 for Python](http://www.h5py.org/)
* [tensorboardX](https://github.com/lanpa/tensorboard-pytorch) (TensorBoard for PyTorch)

### 2. Train SynIQE 
```
python train.py \
  --preprocess True \
  --num_of_layers 4 \
 
```
**NOTE**
* If you've already built the training and validation dataset (i.e. train.h5), set *preprocess* to be False.

### 3. Test
```
python test-3level.py \
  --num_of_layers 4 \
  --log_of_dir /logs \
```

