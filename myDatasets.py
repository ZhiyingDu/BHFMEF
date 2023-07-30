# coding: utf-8
import os
import cv2
import torch
import random
import torch.utils.data as data
import numpy as np

from option import args

def augment_img(img, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)
    '''
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))



class Trainset(data.Dataset):
    def __init__(self, transform):
        super(Trainset, self).__init__()
        self.dir_prefix = args.dir_train
        self.lr_over = os.listdir(self.dir_prefix + 'HR_over/')
        self.lr_over.sort()
        self.lr_under = os.listdir(self.dir_prefix + 'HR_under/')
        self.lr_under.sort()
        self.patch_size = args.patch_size
        self.transform = transform

    def __len__(self):
        assert len(self.lr_over) == len(self.lr_under)
        return len(self.lr_over)
        
    def __getitem__(self, idx):
        img1 = cv2.imread(self.dir_prefix + 'HR_over/' + self.lr_over[idx])
        img6 = cv2.imread(self.dir_prefix + 'HR_under/' + self.lr_under[idx])

        img1YCrCb = cv2.cvtColor(img1, cv2.COLOR_RGB2YCR_CB)
        img6YCrCb = cv2.cvtColor(img6, cv2.COLOR_RGB2YCR_CB)
        # img1YCrCb = cv2.resize(img1YCrCb,(256,256))
        # img6YCrCb = cv2.resize(img6YCrCb,(256,256))
        img1YCrCb, img6YCrCb = self.get_patch(img1YCrCb, img6YCrCb)

        img1Y = img1YCrCb[:, :, 0:1]
        img6Y = img6YCrCb[:, :, 0:1]
        img1Cr = img1YCrCb[:, :, 1:2]
        img1Cb = img1YCrCb[:, :, 2:3]
        img6Cr = img6YCrCb[:, :, 1:2]
        img6Cb = img6YCrCb[:, :, 2:3]

        mode = random.randint(0,7)
        img1Y = augment_img(img1Y, mode=mode)
        img6Y = augment_img(img6Y, mode=mode)
        img1Cr = augment_img(img1Cr, mode=mode)
        img1Cb = augment_img(img1Cb, mode=mode)
        img6Cr = augment_img(img6Cr, mode=mode)
        img6Cb = augment_img(img6Cb, mode=mode)
        # cv2.imwrite("1.png",img1Y)
        # cv2.imwrite("2.png",img6Y)
        # assert 0 == 1
        # cv2.imwrite(0)



        img1Y = img1Y.astype(np.uint8)
        img6Y = img6Y.astype(np.uint8)
        img1Cr = img1Cr.astype(np.uint8)
        img1Cb = img1Cb.astype(np.uint8)
        img6Cr = img6Cr.astype(np.uint8)
        img6Cb = img6Cb.astype(np.uint8)

        img1Y = self.transform(img1Y)
        img1Y = torch.clamp(img1Y, 1/255.0, 254/255.0)

        img6Y = self.transform(img6Y)
        img6Y = torch.clamp(img6Y, 1/255.0, 254/255.0)

        img1Cr = self.transform(img1Cr)
        img1Cb = self.transform(img1Cb)
        img6Cr = self.transform(img6Cr)
        img6Cb = self.transform(img6Cb)

        seq = []
        seq.append(img1Y)
        seq.append(img6Y)
        seq.append(img1Cr)
        seq.append(img1Cb)
        seq.append(img6Cr)
        seq.append(img6Cb)

        return seq
        
    def get_patch(self, l_over, l_under):
        lh, lw = l_over.shape[:2]
        l_stride = self.patch_size
        # scale = self.scale
        # h_stride = l_stride * scale

        x = random.randint(0, lw - l_stride)
        y = random.randint(0, lh - l_stride)
        # ox = scale * x
        # oy = scale * y

        l_over = l_over[y:y + l_stride, x:x + l_stride, :]
        l_under = l_under[y:y + l_stride, x:x + l_stride, :]
        # h_over = h_over[oy:oy + h_stride, ox:ox + h_stride, :]
        # h_under = h_under[oy:oy + h_stride, ox:ox + h_stride, :]
        # h = h[oy:oy + h_stride, ox:ox + h_stride, :]

        return l_over, l_under

# T = Trainset()
# T.__getitem__(0)
class TestData(data.Dataset):
    def __init__(self, transform):
        super(TestData, self).__init__()
        self.dir_prefix = args.dir_test
        self.lr_over = os.listdir(self.dir_prefix + 'testA/')
        self.lr_under = os.listdir(self.dir_prefix + 'testB/')
        self.lr_over.sort()
        self.lr_under.sort()
        self.transform = transform
        # self.scale = args.scale
        # self.patch_size = args.patch_size

    def __len__(self):
        assert len(self.lr_over) == len(self.lr_under)
        return len(self.lr_over)

    def __getitem__(self, idx):
        img1 = cv2.imread(self.dir_prefix + 'testA/' + self.lr_over[idx])
        img6 = cv2.imread(self.dir_prefix + 'testB/' + self.lr_under[idx])

        img1YCrCb = cv2.cvtColor(img1, cv2.COLOR_RGB2YCR_CB)
        img6YCrCb = cv2.cvtColor(img6, cv2.COLOR_RGB2YCR_CB)
        img1Y = img1YCrCb[:, :, 0:1]
        img6Y = img6YCrCb[:, :, 0:1]
        img1Cr = img1YCrCb[:, :, 1:2]
        img1Cb = img1YCrCb[:, :, 2:3]
        img6Cr = img6YCrCb[:, :, 1:2]
        img6Cb = img6YCrCb[:, :, 2:3]



        img1Y = self.transform(img1Y)
        img1Y = torch.clamp(img1Y, 1/255.0, 254/255.0)

        img6Y = self.transform(img6Y)
        img6Y = torch.clamp(img6Y, 1/255.0, 254/255.0)
        
        img1Cr = self.transform(img1Cr)
        img1Cb = self.transform(img1Cb)
        img6Cr = self.transform(img6Cr)
        img6Cb = self.transform(img6Cb)
        img1 = self.transform(img1)
        img6 = self.transform(img6)
        # print(img6Y)
        # print(img6)
        # assert 0 == 1
        seq = []
        seq.append(img1Y)
        seq.append(img6Y)
        seq.append(img1Cr)
        seq.append(img1Cb)
        seq.append(img6Cr)
        seq.append(img6Cb)
        seq.append(img1)
        seq.append(img6)
        return seq
