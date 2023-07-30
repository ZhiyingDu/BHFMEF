import torch
import torch.nn as nn
import torch.nn.functional as F
from option import args
from pytorch_msssim import ssim, ms_ssim

class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()

    def forward(self, image_fused, *tensors):
        img1 = tensors[0]
        img2 = tensors[1]
        img3 = tensors[2]
        img4 = tensors[3]

        intensity_joint = 0.25 * img1 + 0.25 * img2 + 0.25 * img3 + 0.25 * img4
        Loss_intensity = F.l1_loss(image_fused, intensity_joint)
        return Loss_intensity



class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda(int(args.device))
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda(int(args.device))

    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)


class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_fused, *tensors):
        gradient_img1 = self.sobelconv(tensors[0])
        # gradient_A = TF.gaussian_blur(gradient_A, 3, [1, 1])
        gradient_img2 = self.sobelconv(tensors[1])
        gradient_img3 = self.sobelconv(tensors[2])
        gradient_img4 = self.sobelconv(tensors[3])
        # gradient_B = TF.gaussian_blur(gradient_B, 3, [1, 1])
        gradient_fused = self.sobelconv(image_fused)
        gradient_joint = torch.max(torch.max(torch.max(gradient_img1, gradient_img2),gradient_img3),gradient_img4)
        Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
        return Loss_gradient


class L_SSIM(nn.Module):
    def __init__(self):
        super(L_SSIM, self).__init__()

    def forward(self, image_fused, *tensors):
        img1 = tensors[0]
        img2 = tensors[1]
        img3 = tensors[2]
        img4 = tensors[3]
        Loss_SSIM = 0.25 * ms_ssim(img1, image_fused, data_range=1) + 0.25 * ms_ssim(img2, image_fused, data_range=1) + 0.25 * ms_ssim(img3, image_fused, data_range=1) + 0.25 * ms_ssim(img4, image_fused, data_range=1)
        return Loss_SSIM




class fusion_loss_mef(nn.Module):
    def __init__(self):
        super(fusion_loss_mef, self).__init__()
        self.L_Grad = L_Grad()
        self.L_Inten = L_Intensity()
        self.L_SSIM = L_SSIM()


    def forward(self,e , image_fused, *tensors):

        loss_l1 = 10 * self.L_Inten(image_fused, *tensors)
        loss_gradient = 3 * self.L_Grad(image_fused, *tensors)
        loss_SSIM = 2 * (1 - self.L_SSIM(image_fused, *tensors))

        fusion_loss = loss_l1 + loss_SSIM + loss_gradient
        return fusion_loss, loss_l1, loss_SSIM, loss_gradient
