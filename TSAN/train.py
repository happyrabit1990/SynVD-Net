import os
import argparse
from pickle import TUPLE
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
from models import TSAN
from dataset import prepare_data, Dataset
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="TSAN")
parser.add_argument("--preprocess", type=bool, default=True, help='run prepare_data or not')
parser.add_argument('--pretrain', type=str, default='./logs/net-45.pth', help='load pretrain model')
parser.add_argument("--batchSize", type=int, default=32, help="Training batch size")
parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
parser.add_argument("--patchsize", type=int, default=64, help="Training patch size")
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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True)
    # dataset_val = Dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    net = TSAN(inchannels=3, outchannels=3)
    # what is the initial way
    if opt.pretrain != '':
        load_model(net, opt.pretrain)
    else:
        net.apply(weights_init_kaiming)
    model = net.to(device)

    # need to be modifie
    criterion = nn.L1Loss(reduction='sum')
    # Move to GPU
    # device_ids = [0]
    # model = nn.DataParallel(net, device_ids=device_ids).cuda()
    criterion.cuda()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # training
    writer = SummaryWriter(opt.outf)
    step = 0
    for epoch in range(45, opt.epochs):
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
        
            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())
            out_train = model(imgn_train)
            loss = criterion(out_train, img_train) / (img_train.size()[0]*2)
          
            loss.backward()
            optimizer.step()
            # results
            model.eval()
            # res1 = torch.mean(torch.abs(out_train-img_train))
            # res2 = torch.sum(torch.abs(out_train-img_train))/ (img_train.size()[0]*2)
            # print(res1)
            # print(res2)
            out_train = torch.clamp(model(imgn_train), 0., 1.)
            # plt.figure()
            # plt.subplot(121)
            # plt.imshow(out_train[2, :, :, :].detach().cpu().numpy().transpose((1, 2, 0)))
            # plt.subplot(122)
            # plt.imshow(out_train[20, :, :, :].detach().cpu().numpy().transpose((1, 2, 0)))     

            psnr_train = batch_PSNR(out_train, img_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                (epoch+1, i+1, len(loader_train), loss.item(), psnr_train))
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1

        if ((epoch + 1) % 5) == 0:
            torch.save(model.state_dict(), os.path.join(opt.outf, 'net-%d.pth') % (epoch+1))
        # ## the end of each epoch
       

if __name__ == "__main__":
    if opt.preprocess:  
        prepare_data(data_path='data', patch_size=opt.patchsize, stride=32, aug_times=1) 
    main()
