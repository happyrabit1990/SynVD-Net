import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from glob import glob
import matplotlib
import PIL
import matplotlib.pyplot as plt  # plt 用于显示图片

import argparse
from PIL import Image
from models import DnCNN
# from dataset import prepare_testdata, Dataset_test
from utils import *
#matplotlib.use('TkAgg')
#Uncomment the above line if plt.show() does not work

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="SynIQE_Test")
parser.add_argument("--num_of_layers", type=int, default=4, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--prepare_testdata", type=bool, default=False, help='run prepare_data or not')
# parser.add_argument("--test_data", type=str, default='newspaper', help='test on Set12 or Set68')
# parser.add_argument("--test_gtdata", type=str, default='test', help='test on Set12 or Set68')
# parser.add_argument("--test_noisedataL", type=str, default='test_flickerL', help='test on Set12 or Set68')
# parser.add_argument("--test_noisedata", type=str, default='test_flicker', help='test on Set12 or Set68')
# parser.add_argument("--test_noisedataR", type=str, default='test_flickerR', help='test on Set12 or Set68')
parser.add_argument("--save_img", type=int, default=1, help="save image or not")
opt = parser.parse_args()


def normalize(data):
    return data/255.

def SynIQE_infer(test_files_gt, test_files_L, test_files, test_files_R, save_dir, Height, Width):
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
    gt_image = np.zeros((1, 1, Height, Width), dtype="float32")
    noisy_image = np.zeros((1, 3, Height, Width), dtype="float32")

    psnr_test = 0
    # i = 0
    for idx in range(len(test_files)):
        gt_image0 = load_images(test_files_gt[idx]).astype(np.float32) / 255.0
        noisy_image0 = load_images(test_files_L[idx]).astype(np.float32) / 255.0
        noisy_image1 = load_images(test_files[idx]).astype(np.float32) / 255.0
        noisy_image2 = load_images(test_files_R[idx]).astype(np.float32) / 255.0

        gt_image[0:1, 0:1, :, :] =  gt_image0
        noisy_image[0:1, 0:1, :, :] = noisy_image0
        noisy_image[0:1, 1:2, :, :] = noisy_image1
        noisy_image[0:1, 2:3,:, :, ] = noisy_image2
        # noisy_image_Cur = noisy_image1
        INoisy = torch.Tensor(noisy_image)
        ISource = torch.Tensor(gt_image0)
        ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())
        ##Run test
        with torch.no_grad(): # this can save much memory
            denoised_y = (torch.squeeze(INoisy[:,1:2,:,:]-model(INoisy)).cpu().numpy().clip(0, 1)*255.0).astype(np.uint8)

        ##Show results
        # denoised_y1 = np.clip(denoised_y, 0, 255).astype('uint8')
        save_images(os.path.join(save_dir, 'denoised%d.png' % idx), denoised_y)

        ##compute PSNR
        psnr = compt_psnr(denoised_y, gt_image0*255)
        psnr_test += psnr
        print("%s PSNR %f" % (test_files_gt[idx], psnr))

    psnr_test /= len(test_files)
    print("\nPSNR on test data %f" % psnr_test)



if __name__ == "__main__":
     # Build model
    print('Loading model ...\n')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = DnCNN(inchannels=3, outchannels=1, num_of_layers=opt.num_of_layers)
    model = net.to(device)
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net-100.pth')))
   
    Height = 768
    Width = 1024

    # the 3 level test set
    test_files_gt = glob('lovebird1/gt/*.bmp')
    test_files_L = glob('NeighborView/lovebird1/QP34V4/*.bmp')
    test_files = glob( 'lovebird1/textureQP34_depthQP44/*.bmp')
    test_files_R = glob('NeighborView/lovebird1/QP34V6/*.bmp')
    save_file = 'test/3/lovebird1'
    SynIQE_infer(test_files_gt, test_files_L, test_files, test_files_R, save_file, Height, Width)

    Height = 1088
    Width = 1920

    test_files_gt = glob('poznanhall2/gt/*.bmp')
    test_files_L = glob('NeighborView/poznanhall2/QP34V7/*.bmp')
    test_files = glob( 'poznanhall2/textureQP34_depthQP36/*.bmp')
    test_files_R = glob('NeighborView/poznanhall2/QP34V5/*.bmp')
    save_file = 'test/3/poznanhall2'
    SynIQE_infer(test_files_gt, test_files_L, test_files, test_files_R, save_file, Height, Width)


