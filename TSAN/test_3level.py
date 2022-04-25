import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from glob import glob
import cv2
import matplotlib
import PIL
import matplotlib.pyplot as plt  # plt 用于显示图片

import argparse
from PIL import Image
from models import TSAN
# from dataset import prepare_testdata, Dataset_test
from utils import *
#matplotlib.use('TkAgg')
#Uncomment the above line if plt.show() does not work

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="TSAN_Test")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
# parser.add_argument("--prepare_testdata", type=bool, default=False, help='run prepare_data or not')
# parser.add_argument("--test_data", type=str, default='newspaper', help='test on Set12 or Set68')
# parser.add_argument("--test_gtdata", type=str, default='test', help='test on Set12 or Set68')
# parser.add_argument("--test_noisedata", type=str, default='test_flicker', help='test on Set12 or Set68')
# parser.add_argument("--save_img", type=int, default=1, help="save image or not")
opt = parser.parse_args()


def normalize(data):
    return data/255.

def TSAN_infer(test_files_gt, test_files, save_dir, Height, Width):
    # save image
    # if opt.save_img:
    #     save_dir = 'results/saveImg/{}_denoise'.format(opt.test_data)
    #     save_dir_gt = 'results/saveImg/{}_gt'.format(opt.test_data)
    #     save_dir_noise= 'results/saveImg/{}_noise'.format(opt.test_data)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model.eval()
    # load data info
    print('Loading data info ...\n')

    # noisy_image = np.zeros((1, Height, Width, 3), dtype="float32")
    gt_image = np.zeros((1, 3, Height, Width), dtype="float32")
    noisy_image = np.zeros((1, 3, Height, Width), dtype="float32")

    psnr_test = 0
    # i = 0
    for idx in range(len(test_files)):
        gt_image0 = cv2.imread(test_files_gt[idx]).astype(np.float32) / 255.0
        noisy_image0 = cv2.imread(test_files[idx]).astype(np.float32) / 255.0
        gt_image0 = np.transpose(gt_image0,(2,0,1))
        noisy_image0 = np.transpose(noisy_image0,(2,0,1)) 
        gt_image[0:1, :, :, :] =  gt_image0
        noisy_image[0:1, :, :, :] = noisy_image0
        # noisy_image_Cur = noisy_image1
        INoisy = torch.Tensor(noisy_image)
        ISource = torch.Tensor(gt_image)
        ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())
        ##Run test
        with torch.no_grad(): # this can save much memory
            denoised_BGR = (torch.squeeze(model(INoisy)).cpu().numpy().clip(0, 1)*255.0).astype(np.uint8)

        ##Show results
        # denoised_y1 = np.clip(denoised_y, 0, 255).astype('uint8')
       
        ##compute PSNR
        psnr = compt_psnr(denoised_BGR, gt_image*255)
        psnr_test += psnr
        print("%s PSNR %f" % (test_files_gt[idx], psnr))

        denoised_BGR = np.transpose(denoised_BGR,(1,2,0))
        denoised_ycrcb = cv2.cvtColor(denoised_BGR,cv2.COLOR_BGR2YCrCb)
        denoised_y = denoised_ycrcb[:,:,0:1]
        cv2.imwrite(os.path.join(save_dir, 'denoised%d.png' % idx), denoised_y)


    psnr_test /= len(test_files)
    print("\nPSNR on test data %f" % psnr_test)



if __name__ == "__main__":
     # Build model
    print('Loading model ...\n')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = TSAN(inchannels=3, outchannels=3)
    model = net.to(device)
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net-100.pth')))
    
    Height = 768
    Width = 1024

    # the 3 level test set
    # test_files_gt = glob('VVIDatabase_colorImgs/kendo/kendo_ref/*.bmp')
    # test_files = glob( 'VVIDatabase_colorImgs/kendo/kendo_1Dfast_textureQP36_depthQP38/*.bmp')
    # save_file = 'test/3/kendo'
    # TSAN_infer(test_files_gt, test_files, save_file, Height, Width)

    test_files_gt = glob('VVIDatabase_colorImgs/lovebird1/lovebird1_ref/*.bmp')
    test_files = glob( 'VVIDatabase_colorImgs/lovebird1/lovebird1_1Dfast_textureQP34_depthQP44/*.bmp')
    save_file = 'test/3/lovebird1'
    TSAN_infer(test_files_gt, test_files, save_file, Height, Width)

    # test_files_gt = glob('VVIDatabase_colorImgs/newspaper/newspaper_ref/*.bmp')
    # test_files = glob( 'VVIDatabase_colorImgs/newspaper/newspaper_1Dfast_textureQP38_depthQP44/*.bmp')
    # save_file = 'test/3/newspaper'
    # TSAN_infer(test_files_gt, test_files, save_file, Height, Width)

    # test_files_gt = glob('VVIDatabase_colorImgs/outdoor/outdoor_ref/*.bmp')
    # test_files = glob( 'VVIDatabase_colorImgs/outdoor/outdoor_1Dfast_textureQP36_depthQP38/*.bmp')
    # save_file = 'test/3/outdoor'
    # TSAN_infer(test_files_gt, test_files, save_file, Height, Width)

    Height = 1088
    Width = 1920

    test_files_gt = glob('VVIDatabase_colorImgs/poznanhall2/poznanhall2_ref/*.bmp')
    test_files = glob( 'VVIDatabase_colorImgs/poznanhall2/poznanhall2_1Dfast_textureQP34_depthQP36/*.bmp')
    test_files_gt.sort()
    test_files.sort()
    save_file = 'test/3/poznanhall2'
    TSAN_infer(test_files_gt, test_files, save_file, Height, Width)

    # test_files_gt = glob('VVIDatabase_colorImgs/dancer/dancer_ref/*.bmp')
    # test_files = glob( 'VVIDatabase_colorImgs/dancer/dancer_1Dfast_textureQP32_depthQP28/*.bmp')
    # test_files_gt.sort()
    # test_files.sort()
    # save_file = 'test/3/dancer'
    # TSAN_infer(test_files_gt, test_files, save_file, Height, Width)

    # test_files_gt = glob('VVIDatabase_colorImgs/poznancarpark/poznancarpark_ref/*.bmp')
    # test_files = glob('VVIDatabase_colorImgs/poznancarpark/poznancarpark_1Dfast_textureQP34_depthQP36/*.bmp')
    # test_files_gt.sort()
    # test_files.sort()
    # save_file = 'test/3/poznancarpark'
    # TSAN_infer(test_files_gt, test_files, save_file, Height, Width)


    # test_files_flicker = glob( 'E:/CNNtest/DnCNN-syn-1/matlab_prepareInput/extractYImgs/lovebird1/_1Dfast_textureQP34_depthQP44/*.bmp')
    # save_file = 'F:/E-CNNtest-resume/DnCNN-syn-1-nei-all-general/test_unet/3/lovebird1'
    # TSAN_infer(test_files_gt, test_files_L, test_files, test_files_R, save_file, Height, Width)

    # test_files_flicker = glob( 'F:/E-CNNtest-resume/matlab_prepareInput/extractYImgs/outdoor_atm/_1Dfast_textureQP36_depthQP38/*.bmp')
    # save_file = 'F:/E-CNNtest-resume/DnCNN-syn-1-nei-all-general/test_unet/3/outdoor'
    # TSAN_infer(test_files_gt, test_files_L, test_files, test_files_R, save_file, Height, Width)
    
