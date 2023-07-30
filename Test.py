from myDatasets import TestData
from option import args
from torchvision import transforms
from torch.autograd import Variable
# from option import args
from models.MEF_Model import SwinFusion
from models.MEF_Model import SwinFusion
from models.GC_model import GC
from models.network_ffdnet import FFDNet as net
import torch.utils.data as data
import torch
import numpy as np
# import torch.nn as nn
import cv2 as cv
from color_enhance import CE

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0], std=[1])])
test_set = TestData(transform=transform)
args.batch_size = 1
test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

device = torch.device("cuda:" + args.device)

model = SwinFusion()
model.to(device)
state = torch.load(args.model_path + args.model,map_location={'cuda:0': 'cuda:0'})
# state = torch.load(args.model_path + args.model)
model.load_state_dict(state['model'])

# device = torch.device("cuda:" + '0')
#GC_model
GC_model = GC()
GC_model.to(device)
GC_state = torch.load(args.model_path + args.GC_model,map_location={'cuda:0': 'cuda:0'})
GC_model.load_state_dict(GC_state['model'])



Denoise_model = net(in_nc=1, out_nc=1, nc=64, nb=15, act_mode='R')
Denoise_state = torch.load(args.model_path + args.Denoise_model,map_location={'cuda:1': 'cuda:0'})
Denoise_model.load_state_dict(Denoise_state, strict=True)
for k, v in Denoise_model.named_parameters():
    v.requires_grad = False
Denoise_model = Denoise_model.to(device)
noise_level_img = 10

EPS = 1e-8

with torch.no_grad():
    k = -1
    for imgs in test_loader:
        k += 1
        model.eval()
        # if k+1<72:
        #     continue
        print('Processing picture No.{}'.format(k + 1))
        b, c, h, w = imgs[0].size()
        vimg = []
        _, _, h_old, w_old = imgs[0].size()
        h_pad = 0
        w_pad = 0
        if h_old % args.Window_size != 0:
            h_pad = (h_old // args.Window_size + 1) * args.Window_size - h_old
        if w_old % args.Window_size != 0:
            w_pad = (w_old // args.Window_size + 1) * args.Window_size - w_old

        weight = []
        for i in range(2):
            img = Variable(imgs[i].type(torch.FloatTensor).to(device))
            img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h_old + h_pad, :]
            img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w_old + w_pad]
            vimg.append(img)


        # vimgs = vimgs.type(torch.FloatTensor)
        img2_mid, img2, map = GC_model(vimg[0])
        for i in range(args.batch_size):
            bias = (torch.randn(img2[i:i+1,:,:,:].shape) * (noise_level_img / 255.)).to(device)
            img2[i:i+1,:,:,:] += bias
            sigma = torch.full((1,1,1,1), noise_level_img/255.).type_as(img2[i:i+1,:,:,:])
            img2[i:i+1,:,:,:] = Denoise_model(img2[i:i+1,:,:,:], sigma)

        img3mid, img3, map = GC_model(vimg[1])
        for i in range(args.batch_size):
            bias = (torch.randn(img3[i:i+1,:,:,:].shape) * (noise_level_img / 255.)).to(device)
            img3[i:i+1,:,:,:] += bias
            sigma = torch.full((1,1,1,1), noise_level_img/255.).type_as(img3[i:i+1,:,:,:])
            img3[i:i+1,:,:,:] = Denoise_model(img3[i:i+1,:,:,:], sigma)


        vimgs = []
        vimgs.append(vimg[0])
        vimgs.append(img2)
        vimgs.append(img3)
        vimgs.append(vimg[1])


        vimg = vimgs.copy()
        vres, init_Y, edge1, edge2 = model(*vimg)
        vres = vres[..., :h_old, :w_old]


        Cbs = torch.zeros(args.batch_size,2, h, w)
        Crs = torch.zeros(args.batch_size,2, h, w)

        Crs[:,0:1,:,:] = imgs[2]
        Cbs[:,0:1,:,:] = imgs[3]
        Crs[:,1:2,:,:] = imgs[4]
        Cbs[:,1:2,:,:] = imgs[5]
        img_cr = torch.cat((Crs[:,0:1,:,:], Crs[:,1:2,:,:]), dim=0).to(device)
        img_cb = torch.cat((Cbs[:,0:1,:,:], Cbs[:,1:2,:,:]), dim=0).to(device)
        w_cr = (torch.abs(img_cr-0.5) + EPS) / torch.sum(torch.abs(img_cr-0.5) + EPS, dim=0)
        w_cb = (torch.abs(img_cb-0.5) + EPS) / torch.sum(torch.abs(img_cb-0.5) + EPS, dim=0)

        fused_img_cr = torch.sum(w_cr * img_cr, dim=0, keepdim=True).clamp(-1, 1)
        fused_img_cb = torch.sum(w_cb * img_cb, dim=0, keepdim=True).clamp(-1, 1)


        fused_img = torch.cat((vres, fused_img_cr, fused_img_cb), dim=1)
        fused_img = fused_img * 255
        fused_img = fused_img.squeeze(0)
        fused_img = fused_img.cpu().numpy()
        fused_img = np.transpose(fused_img, (1, 2, 0))
        fused_img = fused_img.astype(np.uint8)
        fused_img = cv.cvtColor(fused_img, cv.COLOR_YCrCb2RGB)

        img_new = CE(fused_img, 0.2)
        img_new = img_new * 255.0
        img_new = img_new.astype(fused_img.dtype)

        cv.imwrite((args.save_dir + str(k + 201) + ".png"), img_new)
