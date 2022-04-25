import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from model import UNet
from dataset import prepare_data, Dataset
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="SynVD-Unet")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument('--pretrain', type=str, default='', help='load pretrain model')
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
# parser.add_argument('--bilinear', type=bool, default=False, help='')
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--fw", type=float, default=3, help="flicker weight")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
opt = parser.parse_args()

def load_model(model, f):
    with open(f, 'rb') as f:
        # pretrained_dict = torch.load(f)
        pretrained_dict = torch.load(f, map_location=lambda storage, loc: storage.cuda(0))
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

def freeze(model):
    for p in model.parameters():
        p.requires_grad = False

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad = True
        
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True)
    # dataset_val = Dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    # net = UNet(in_ch=1,out_ch=1)
    net = UNet(in_ch=1, out_ch=1)
    if opt.pretrain != '':
        load_model(net, opt.pretrain)
    else:
        net.apply(weights_init_kaiming)
    model = net.to(device)
    criterion = nn.MSELoss(size_average=False)
    # Move to GPU
    # device_ids = [0]
    # model = nn.DataParallel(net, device_ids=device_ids).cuda()
    criterion.cuda()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # training
    writer = SummaryWriter(opt.outf)
    step = 0
    for epoch in range(0, opt.epochs):
        current_lr = opt.lr
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # train
        for i, data in enumerate(loader_train, 0):
            # training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_train = data[0]
            imgn_train = data[1]
            img_train0 = img_train[:, 0:1, :, :]
            img_train1 = img_train[:, 1:2, :, :]
            img_train2 = img_train[:, 2:3, :, :]
            imgn_train0 = imgn_train[:, 0:1, :, :]
            imgn_train1 = imgn_train[:, 1:2, :, :]
            imgn_train2 = imgn_train[:, 2:3, :, :]
            # noise1 = imgn_train1- img_train1

            img_train0, img_train1, img_train2, imgn_train0, imgn_train1, imgn_train2, = Variable(img_train0.cuda()), Variable(img_train1.cuda()), Variable(img_train2.cuda()),\
                                                                                          Variable(imgn_train0.cuda()), Variable(imgn_train1.cuda()), Variable(imgn_train2.cuda()),
            # noise1 = Variable(noise1.cuda())
    
            out_train1 = model(imgn_train1)
            # model.eval()
            # freeze(model)
            out_train0 = model(imgn_train0)
            out_train2 = model(imgn_train2)
            # model.train()
            # unfreeze(model)
            # loss = 1/ (imgn_train1.size()[0]*2)*(criterion(out_train1, noise1))

            loss = 1/ (imgn_train1.size()[0]*2)*(criterion(img_train1, out_train1) + opt.fw * criterion((img_train1 - img_train0),(out_train1 - out_train0)) \
                        + opt.fw * criterion((img_train1 - img_train2), (out_train1 - out_train2)))

            loss.backward()
            optimizer.step()
            # results
            model.eval()
            out_train = torch.clamp(out_train1, 0., 1.)
            psnr_train = batch_PSNR(out_train, img_train1, 1.)
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                (epoch+1, i+1, len(loader_train), loss.item(), psnr_train))
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1

        if ((epoch + 1) % 1) == 0:
            torch.save(model.state_dict(), os.path.join(opt.outf, 'net-%d.pth') % (epoch+1))
        # ## the end of each epoch
        # model.eval()
        # # validate
        # psnr_val = 0
        # for k in range(len(dataset_val)):
        #     img_val = torch.unsqueeze(dataset_val[k], 0)
        #     noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.val_noiseL/255.)
        #     imgn_val = img_val + noise
        #     img_val, imgn_val = Variable(img_val.cuda(), volatile=True), Variable(imgn_val.cuda(), volatile=True)
        #     out_val = torch.clamp(imgn_val-model(imgn_val), 0., 1.)
        #     psnr_val += batch_PSNR(out_val, img_val, 1.)
        # psnr_val /= len(dataset_val)
        # print("\n[epoch %d] PSNR_val: %.4f" % (epoch+1, psnr_val))
        # writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        # # log the images
        # out_train = torch.clamp(imgn_train-model(imgn_train), 0., 1.)
        # Img = utils.make_grid(img_train.data, nrow=8, normalize=True, scale_each=True)
        # Imgn = utils.make_grid(imgn_train.data, nrow=8, normalize=True, scale_each=True)
        # Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        # writer.add_image('clean image', Img, epoch)
        # writer.add_image('noisy image', Imgn, epoch)
        # writer.add_image('reconstructed image', Irecon, epoch)
        # save model
        
    
if __name__ == "__main__":
    if opt.preprocess:  
        prepare_data(data_path='data', patch_size=32, stride=24, aug_times=1) 
    main()
