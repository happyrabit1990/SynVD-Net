import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import glob
import torch.utils.data as udata
from utils import data_augmentation

def normalize(data):
    return data/255.

# def Im2Patch(img, win, stride=1):
#     k = 0
#     endc = img.shape[0]
#     endw = img.shape[1]
#     endh = img.shape[2]
#     patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
#     TotalPatNum = patch.shape[1] * patch.shape[2]
#     # Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
#     Y = np.zeros([TotalPatNum, endc, win*win], np.float32)
#     for i in range(win):
#         for j in range(win):
#             patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
#             Y[:,:,k] = np.array(patch[:]).reshape(TotalPatNum, endc)
#             k = k + 1
#     return Y.reshape([TotalPatNum, endc, win, win])

def Im2Patch_3c(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    # Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    Y1 = np.zeros([TotalPatNum, 1, win*win], np.float32)
    Y2 = np.zeros([TotalPatNum, 1, win*win], np.float32)
    Y3 = np.zeros([TotalPatNum, 1, win*win], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[0,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y1[:,:,k] = np.array(patch[:]).reshape(TotalPatNum, 1)
            patch = img[1,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y2[:,:,k] = np.array(patch[:]).reshape(TotalPatNum, 1)
            patch = img[2,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y3[:,:,k] = np.array(patch[:]).reshape(TotalPatNum, 1)
            k = k + 1
    Y = np.concatenate((Y1,Y2,Y3),axis=1)
    return Y.reshape([TotalPatNum, endc, win, win])

def prepare_data(data_path, patch_size, stride, aug_times=1):
    # train
    print('process training data')
    files_gt = glob.glob(os.path.join(data_path, 'training-gt', '*.bmp'))
    files_noise = glob.glob(os.path.join(data_path, 'training-input-3level', '*.bmp'))
    files_gt.sort()
    files_noise.sort()
    # h5f = h5py.File('train.h5', 'w')
    train_num = 0
    channel = 3
    for i in range(len(files_gt)):
        if i == 0:
            h5f = h5py.File("train.h5", "w") #build File object
            x = h5f.create_dataset("gt_train", (1, channel, patch_size,patch_size), 
            maxshape=(None,channel, patch_size,patch_size), dtype =np.float32)# build gt_train dataset
            y = h5f.create_dataset("noise_train", (1, channel, patch_size,patch_size), 
            maxshape=(None,channel, patch_size,patch_size), dtype =np.float32)# build noise_train dataset
        else:
            h5f = h5py.File("train.h5", "a") # add mode
            x = h5f["gt_train"]
            y = h5f["noise_train"]

        img_gt = cv2.imread(files_gt[i])       
        h, w, c = img_gt.shape
        # Img_gt_g = np.zeros((c, w, h), dtype="float32")
        # Img_no_g = np.zeros((c, w, h), dtype="float32")

        # img_gt = cv2.imread(files_gt[i])
        img_gt = np.transpose(img_gt,(2,0,1))
        img_gt = np.float32(normalize(img_gt)) 
        img_no = cv2.imread(files_noise[i])
        img_no = np.transpose(img_no,(2,0,1))
        img_no = np.float32(normalize(img_no))  

     
        patches_gt = Im2Patch_3c(img_gt, win=patch_size, stride=stride)
        patches_no = Im2Patch_3c(img_no, win=patch_size, stride=stride)

        print("file: %s  # samples: %d" % (files_gt[i], patches_gt.shape[0]*aug_times))
        print("file: %s  # samples: %d" % (files_noise[i], patches_no.shape[0]*aug_times))

        for n in range(patches_gt.shape[0]):
            data_gt = patches_gt[n,:,:,:].copy()
            data_no = patches_no[n,:,:,:].copy()
            idx = np.random.randint(1,8)
            data_gt_aug = data_augmentation(data_gt, idx)
            data_no_aug = data_augmentation(data_no, idx)
            patches_gt[n,:,:,:] = data_gt_aug
            patches_no[n,:,:,:] = data_no_aug

            # plt.figure()
            # plt.imshow(patches_gt[n,:,:,:].transpose((1, 2, 0)))
            # plt.show()          
            # plt.figure()
            # plt.imshow(patches_no[n,:,:,:].transpose((1, 2, 0)))
            # plt.show()

        x.resize([train_num + patches_gt.shape[0]*aug_times,channel, patch_size,patch_size])
        y.resize([train_num + patches_no.shape[0]*aug_times,channel, patch_size,patch_size])
        x[train_num: train_num + patches_gt.shape[0]*aug_times] = patches_gt
        y[train_num: train_num + patches_no.shape[0]*aug_times] = patches_no 

        train_num += patches_gt.shape[0]*aug_times

    h5f.close()
    print('training set, # samples %d\n' % train_num)
    files_gt.clear()
    files_noise.clear()
  

# def prepare_data(data_path, patch_size, stride, aug_times=1):
#     # train
#     print('process training data')
#     # scales = [1, 0.9, 0.8, 0.7]
#     scales = [1]
#     files_gt = glob.glob(os.path.join(data_path, 'training-gt', '*.bmp'))
#     files_noise = glob.glob(os.path.join(data_path, 'training-input-3level', '*.bmp'))
#     files_gt.sort()
#     files_noise.sort()
#     # h5f = h5py.File('train.h5', 'w')
#     train_num = 0
#     frame_len = 1
#     channel = 3
    
#     for k in range(len(scales)):
#         for i in range(0, len(files_gt)):
#             if i == 0:
#                 h5f = h5py.File("train.h5", "w") #build File object
#                 x = h5f.create_dataset("gt_train", (1, channel, patch_size,patch_size), 
#                 maxshape=(None,channel, patch_size,patch_size), dtype =np.float32)# build gt_train dataset
#                 y = h5f.create_dataset("noise_train", (1, channel, patch_size,patch_size), 
#                 maxshape=(None,channel, patch_size,patch_size), dtype =np.float32)# build noise_train dataset
#             else:
#                 h5f = h5py.File("train.h5", "a") # add mode
#                 x = h5f["gt_train"]
#                 y = h5f["noise_train"]
           
#              #写入数据集 

#             # print('{} images are dealed with'.format(img_i))

#             # img = Image.open(files_gt[i])# convert RGB to gray
#             img = cv2.imread(files_gt[i])
#             h, w, c = img.shape
#             # cv2.namedWindow('image', cv2.WINDOW_NORMAL)#
#             # cv2.imshow('image',img)#
#             # cv2.waitKey(0)#
#             # cv2.destroyAllWindows()

#             newsize = (int(h*scales[k]), int(w*scales[k]))
#             Img_gt_g = np.zeros((c, int(w*scales[k]), int(h*scales[k])), dtype="float32")
#             Img_no_g = np.zeros((c, int(w*scales[k]), int(h*scales[k])), dtype="float32")
#             for j in range(frame_len):
#                 idx = i
#                 img_gt = Image.open(files_gt[idx]).convert('L')  # convert RGB to gray 
#                 img_gt = img_gt.resize(newsize, resample=PIL.Image.BICUBIC) 
#                 img_gt = np.reshape(np.array(img_gt), (c, img_gt.size[1], img_gt.size[0]))  # extend one dimension
#                 img_no = Image.open(files_noise[idx]).convert('L')  # convert RGB to gray 
#                 img_no = img_no.resize(newsize, resample=PIL.Image.BICUBIC) 
#                 img_no = np.reshape(np.array(img_no), (c, img_no.size[1], img_no.size[0]))  # extend one dimension
#                 # Img = np.expand_dims(Img[:,:,0].copy(), 0)
#                 img_gt = np.float32(normalize(img_gt))
#                 img_no = np.float32(normalize(img_no))
#                 Img_gt_g [j,:,:]= img_gt
#                 Img_no_g [j,:,:]= img_no
                 

#             patches_gt = Im2Patch_3c(Img_gt_g, win=patch_size, stride=stride)
#             patches_no = Im2Patch_3c(Img_no_g, win=patch_size, stride=stride)

#             print("file: %s scale %.1f # samples: %d" % (files_gt[i*frame_len], scales[k], patches_gt.shape[0]*aug_times))
#             print("file: %s scale %.1f # samples: %d" % (files_noise[i*frame_len], scales[k], patches_no.shape[0]*aug_times))

#             x.resize([train_num + patches_gt.shape[0]*aug_times,channel, patch_size,patch_size])
#             y.resize([train_num + patches_no.shape[0]*aug_times,channel, patch_size,patch_size])

#             x[train_num: train_num + patches_gt.shape[0]*aug_times] = patches_gt
#             y[train_num: train_num + patches_no.shape[0]*aug_times] = patches_no

#             # tt = patches_gt[20,0,:,:]
#             # tt1 = patches_no[20,1,:,:]
#             # zz = tt.reshape(32,32)
#             # zz1 = tt1.reshape(32,32)
#             # plt.imshow(zz, cmap='Greys_r')
#             # # plt.axis('off')  # 
#             # plt.show()
#             # plt.imshow(zz1, cmap='Greys_r')
#             # # plt.axis('off')  # 
#             # plt.show()


#             train_num += patches_gt.shape[0]*aug_times
#             # for n in range(patches_gt.shape[3]):
#             #     data = patches_gt[:,:,:,n].copy()
#             #     h5f.create_dataset(str(train_num), data=data)
#             #     train_num += 1
#             #     for m in range(aug_times-1):
#             #         data_aug = data_augmentation(data, np.random.randint(1,8))
#             #         h5f.create_dataset(str(train_num)+"_aug_%d" % (m+1), data=data_aug)
#             #         train_num += 1
#     h5f.close()
#     files_gt.clear()
#     files_noise.clear()
#     # val
#     print('\nprocess validation data')
#     view_num = 3
#     frame_len = 1
#     # files_L = glob.glob(os.path.join(data_path, 'test_flickerL', '*.bmp'))
#     # files = glob.glob(os.path.join(data_path, 'test_flicker', '*.bmp'))
#     # files_R = glob.glob(os.path.join(data_path, 'test_flickerR', '*.bmp'))
#     # files.sort()
#     # h5f = h5py.File('val.h5', 'w')
#     # val_num = 0
#     # for i in range( len(files)//frame_len):
#     #     img = Image.open(files[i*frame_len]).convert('L')  # convert RGB to gray
#     #     h, w = img.size
#     #     Img_g = np.zeros((frame_len*view_num, w, h), dtype="float32")
#     #     for j in range(frame_len):
#     #         idx = i*frame_len + j
#     #         print("file: %s" % files[idx])
#     #         imgL = Image.open(files_L[idx]).convert('L')  # convert RGB to gray
#     #         imgL = np.reshape(np.array(imgL), (1, imgL.size[1], imgL.size[0]))
#     #         imgL = np.float32(normalize(imgL))
#     #         img = Image.open(files[idx]).convert('L')  # convert RGB to gray
#     #         img = np.reshape(np.array(img), (1, img.size[1], img.size[0]))
#     #         img = np.float32(normalize(img))
#     #         imgR = Image.open(files_R[idx]).convert('L')  # convert RGB to gray
#     #         imgR = np.reshape(np.array(imgR), (1, imgR.size[1], imgR.size[0]))
#     #         imgR = np.float32(normalize(imgR))
#     #         Img_g [j,:,:]= imgL
#     #         Img_g [j+1,:,:]= img
#     #         Img_g [j+2,:,:]= imgR
#     #     h5f.create_dataset(str(val_num), data=Img_g)
#     #     val_num += 1
#     # h5f.close()
#     print('training set, # samples %d\n' % train_num)
#     # print('val set, # samples %d\n' % val_num)

def prepare_testdata(data_path):
    print('\nprocess test data')
    files_gt = glob.glob(os.path.join(data_path, 'newspaper_gt', '*.bmp'))
    files_noise = glob.glob(os.path.join(data_path, 'newspaper_noise', '*.bmp'))
    files_gt.sort()
    files_noise.sort()


    # h5f = h5py.File('test.h5', 'w')
    frame_len = 1
    test_num = 0

    img = Image.open(files_gt[0]).convert('L')  # convert RGB to gray
    h, w = img.size
    # Img_g = np.zeros((frame_len, w, h), dtype="float32")

    for i in range( len(files_gt)//frame_len):
        if i == 0:
            h5f = h5py.File("test.h5", "w") #build File object
            x = h5f.create_dataset("gt_test", (1, frame_len, img.size[1],img.size[0]), 
            maxshape=(None,frame_len, img.size[1],img.size[0]), dtype =np.float32)# build gt_train dataset
            y = h5f.create_dataset("noise_test", (1, frame_len, img.size[1],img.size[0]), 
            maxshape=(None,frame_len, img.size[1],img.size[0]), dtype =np.float32)# build noise_train dataset
        else:
            h5f = h5py.File("test.h5", "a") # add mode
            x = h5f["gt_test"]
            y = h5f["noise_test"]

        # img = Image.open(files[i*frame_len]).convert('L')  # convert RGB to gray
        # h, w = img.size
        Img_gt_g = np.zeros((frame_len, w, h), dtype="float32")
        Img_no_g = np.zeros((frame_len, w, h), dtype="float32")
        for j in range(frame_len):
            idx = i*frame_len + j
            print("file: %s" % files_gt[idx])
            img_gt = Image.open(files_gt[idx]).convert('L')  # convert RGB to gray
            img_gt = np.reshape(np.array(img_gt), (1, img_gt.size[1], img_gt.size[0]))
            img_gt = np.float32(normalize(img_gt))
            img_no = Image.open(files_noise[idx]).convert('L')  # convert RGB to gray
            img_no = np.reshape(np.array(img_no), (1, img_no.size[1], img_no.size[0]))
            img_no = np.float32(normalize(img_no))
            Img_gt_g [j,:,:]= img_gt
            Img_no_g [j,:,:]= img_no

        x.resize([test_num + 1,frame_len, w, h])
        y.resize([test_num + 1,frame_len, w, h])
        x[test_num: test_num + 1] = Img_gt_g
        y[test_num: test_num + 1] = Img_no_g
        test_num += 1

    h5f.create_dataset("num", data=test_num)    
    h5f.close()
    print('test set, # samples %d\n' % test_num)

class Dataset(udata.Dataset):
    def __init__(self, train=True):
        super(Dataset, self).__init__()
        self.train = train
        if self.train:
            h5f = h5py.File('train.h5', 'r')
        else:
            h5f = h5py.File('test.h5', 'r')
        n1 = h5f.get('gt_train')
        # self.len =  np.array(n1).shape[0]
        # self.len =  np.array(n1).shape[0]
        self.len =  138210
        # random.shuffle(self.keys)
        h5f.close()
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        if self.train:
            h5f = h5py.File('train.h5', 'r')
        else:
            h5f = h5py.File('test.h5', 'r')
        # gt_data = h5f['gt_train'].value[index:index+1]
        # noise_data = h5f['noise_train'].value[index:index+1]
        gt_data = torch.Tensor(np.array(h5f['gt_train'][index]))
        noise_data = torch.Tensor(np.array(h5f['noise_train'][index]))
        # gt_data =  torch.Tensor(np.array(h5f.get('gt_train')))
        # noise_data = torch.Tensor(np.array(h5f.get('noise_train')))
        h5f.close()
        return gt_data,noise_data

class Dataset_test(udata.Dataset):
    def __init__(self, train=True):
        super(Dataset_test, self).__init__()
        self.train = train
        if self.train:
            h5f = h5py.File('train.h5', 'r')
        else:
            h5f = h5py.File('test.h5', 'r')
        n1 = h5f.get('num')
        # self.len =  np.array(n1).shape[0]
        self.len =  np.array(n1)
        # random.shuffle(self.keys)
        h5f.close()
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        if self.train:
            h5f = h5py.File('train.h5', 'r')
        else:
            h5f = h5py.File('test.h5', 'r')
    
        gt_data = torch.Tensor(np.array(h5f['gt_test'][index]))
        noise_data = torch.Tensor(np.array(h5f['noise_test'][index]))
        h5f.close()
        return gt_data,noise_data


# class Dataset(udata.Dataset):
#     def __init__(self, train=True):
#         super(Dataset, self).__init__()
#         self.train = train
#         if self.train:
#             h5f = h5py.File('train.h5', 'r')
#         else:
#             h5f = h5py.File('val.h5', 'r')
#         self.keys = list(h5f.keys())
#         random.shuffle(self.keys)
#         h5f.close()
#     def __len__(self):
#         return len(self.keys)
#     def __getitem__(self, index):
#         if self.train:
#             h5f = h5py.File('train.h5', 'r')
#         else:
#             h5f = h5py.File('val.h5', 'r')
#         key = self.keys[index]
#         data = np.array(h5f[key])
#         h5f.close()
#         return torch.Tensor(data)
