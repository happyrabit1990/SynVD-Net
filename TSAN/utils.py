import math
import torch
import torch.nn as nn
import numpy as np
from skimage.measure.simple_metrics import compare_psnr

# def weights_init_kaiming(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
#     elif classname.find('Linear') != -1:
#         nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
#     elif classname.find('BatchNorm') != -1:
#         # nn.init.uniform(m.weight.data, 1.0, 0.02)
#         m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
#         nn.init.constant(m.bias.data, 0.0)

def weights_init_kaiming(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(mean=0, std=math.sqrt(2. / 9. / 64.)).clamp_(-0.025, 0.025)
            nn.init.constant(m.bias.data, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')

def weights_init(m):
    for name, param in m.named_parameters():
        if name.find('bias') != -1:
            nn.init.constant_(param, val=0)
        elif name.find('weight') != -1:
            nn.init.kaiming_normal(param, a=0, mode='fan_in')

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])
# def batch_PSNR(img, imclean, data_range):
#     # Img = img.data.cpu().numpy().astype(np.float32)
#     # Iclean = imclean.data.cpu().numpy().astype(np.float32)
#     Img = img.astype(np.float32)
#     Iclean = imclean.astype(np.float32)
#     PSNR = compare_psnr(Iclean, Img, data_range=data_range)
#     return PSNR
#     # for i in range(Img.shape[0]):
#     #     # PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
#     #     PSNR += compare_psnr(Iclean, Img, data_range=data_range)
#     # return (PSNR/Img.shape[0])

def compt_psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    
    if mse > 1000:
        return -100
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def data_augmentation(image, mode):
    out = np.transpose(image, (1,2,0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2,0,1))
