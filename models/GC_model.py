import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class ConvLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        # self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        # print(x.size())
        return F.leaky_relu(self.conv(x), negative_slope=0.2)


class GC(nn.Module):
    def __init__(self):
        super(GC, self).__init__()
        # self.fuse_scheme = fuse_scheme # MAX, MEAN, SUM
        self.conv11 = ConvLeakyRelu2d(1, 32)
        self.conv12 = ConvLeakyRelu2d(32, 32)
        self.conv21 = ConvLeakyRelu2d(32, 32)
        self.conv22 = ConvLeakyRelu2d(32, 32)
        self.conv3 = ConvLeakyRelu2d(32, 8)
        # self.layer1 = nn.Linear(16*num,num)
        # self.layer2 = nn.Linear(num,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, tensors):

        img = tensors
        x = tensors
        fea = self.conv11(img)
        fea = self.conv12(fea)
        fea = self.conv21(fea)
        fea = self.conv22(fea)
        fea = self.conv3(fea)

        fea = self.sigmoid(fea) + 0.8
        # num = torch.mean(img)

        r1,r2,r3,r4,r5,r6,r7,r8 = torch.split(fea, 1, dim=1)


		
        x = x + x - x ** r1
		
        x = x + x - x ** r2
		
        x = x + x - x ** r3
		
        enhance_image_1 = x + x - x ** r4
        x = enhance_image_1 + enhance_image_1 - enhance_image_1 ** r5	
		
        x = x + x - x ** r6
		
        x = x + x - x ** r7
		
        enhance_image = x + x - x ** r8
        r = torch.cat([r1,r2,r3,r4,r5,r6,r7,r8],1)

        return enhance_image_1,enhance_image,r