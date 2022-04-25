import math
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from skimage.measure.simple_metrics import compare_psnr

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant(m.bias.data, 0.0)

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
    Img = img1.astype(np.float32)
    Iclean = img2.astype(np.float32)
    mse = np.mean( (Img - Iclean) ** 2 )
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


def load_images(filelist):
    # pixel value range 0-255
    if not isinstance(filelist, list):
        im = Image.open(filelist).convert('L')
        return np.array(im).reshape(1, 1,im.size[1], im.size[0] )
    data = []
    for file in filelist:
        im = Image.open(file).convert('L')
        data.append(np.array(im).reshape(1, 1,im.size[1], im.size[0] ))
    return data


def save_images(filepath, ground_truth, noisy_image=None, clean_image=None):
    # assert the pixel value range is 0-255
    ground_truth = np.squeeze(ground_truth)
    noisy_image = np.squeeze(noisy_image)
    clean_image = np.squeeze(clean_image)
    if not clean_image.any():
        cat_image = ground_truth
    else:
        cat_image = np.concatenate([ground_truth, noisy_image, clean_image], axis=1)
    im = Image.fromarray(cat_image.astype('uint8')).convert('L')
    im.save(filepath, 'png')
