# RDN-PyTorch

This is a PyTorch implementation of the TIP2017 paper [**](http://ieeexplore.ieee.org/document/7839189/). The author's [MATLAB implementation is here]().

****
This code was written with 

****

## How to run

### 1. Dependences
* [PyTorch](http://pytorch.org/)(<0.4)
* [torchvision](https://github.com/pytorch/vision)
* OpenCV for Python
* [HDF5 for Python](http://www.h5py.org/)
* [tensorboardX](https://github.com/lanpa/tensorboard-pytorch) (TensorBoard for PyTorch)

### 2. Train 
```
python train.py \
  --preprocess True \
  
```
**NOTE**
* If you've already built the training and validation dataset (i.e. train.h5 & val.h5 files), set *preprocess* to be False.

```
**NOTE**
* If you've already built the training and validation dataset (i.e. train.h5 & val.h5 files), set *preprocess* to be False.


### 4. Test
```
python test.py \
  --num_of_layers 17 \
  --logdir logs/ \
 
```
**NOTE**
* *test_data* can be 

## Test Results

### BSD68 Average RSNR

| Noise Level | DnCNN-S | DnCNN-B | DnCNN-S-PyTorch | DnCNN-B-PyTorch |
|:-----------:|:-------:|:-------:|:---------------:|:---------------:|
|     15      |  31.73  |  31.61  |      31.71      |      31.60      |
|     25      |  29.23  |  29.16  |      29.21      |      29.15      |
|     50      |  26.23  |  26.23  |      26.22      |      26.20      |


## Tricks useful for boosting performance
* Parameter initialization:  
Use *kaiming_normal* initialization for *Conv*; Pay attention to the initialization of *BatchNorm*
```
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant(m.bias.data, 0.0)
```
* The definition of loss function  
Set *size_average* to be False when defining the loss function. When *size_average=True*, the **pixel-wise average** will be computed, but what we need is **sample-wise average**.
```
criterion = nn.MSELoss(size_average=False)
```
The computation of loss will be like:
```
loss = criterion(out_train, noise) / (imgn_train.size()[0]*2)
```
where we divide the sum over one batch of samples by *2N*, with *N* being # samples.
