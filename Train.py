import torch
from torch import optim
from myDatasets import Trainset
from models.MEF_Model import SwinFusion
from models.GC_model import GC
from models.network_ffdnet import FFDNet as net
import os 
from torch.autograd import Variable
import numpy as np
import torch.utils.data as data
from option import args
import matplotlib.pyplot as plt
import matplotlib
from MEFLoss import fusion_loss_mef
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from utils.weight import CalWeight_Train

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0], std=[1])])
train_set = Trainset(transform=transform)
train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)


device = torch.device("cuda:" + args.device)
model = SwinFusion()
model.to(device)

#GC_model
GC_model = GC()
GC_model.to(device)
GC_state = torch.load(args.model_path + args.GC_model,map_location={'cuda:0': 'cuda:0'})
GC_model.load_state_dict(GC_state['model'])
# state = torch.load(args.model_path + args.model)

# model.load_state_dict(state['model'])
Denoise_model = net(in_nc=1, out_nc=1, nc=64, nb=15, act_mode='R')
Denoise_state = torch.load(args.model_path + args.Denoise_model,map_location={'cuda:1': 'cuda:0'})
Denoise_model.load_state_dict(Denoise_state, strict=True)
for k, v in Denoise_model.named_parameters():
    v.requires_grad = False
Denoise_model = Denoise_model.to(device)
noise_level_img = 10   

Loss_Fun = fusion_loss_mef().to(device)


optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
scheduler = CosineAnnealingLR(optimizer, args.epoch)

Loss_list = []

EPS = 1e-8
for e in range(args.epoch):
    loss_list = []
    print(e)
    for imgs in train_loader:
        GC_model.eval()
        # Denoise_model.eval()
        b, c, h, w = imgs[0].size()
        Ys = []
        for i in range(2):
            Ys.append(imgs[i])

        optimizer.zero_grad()
        with torch.no_grad():
            vimg = []
            for idx, img in enumerate(Ys):
                vimg.append(Variable(img.type(torch.FloatTensor).to(device)))


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
            # print(img5.shape)
            # assert img5.shape == vimg[1].shape
            vimgs = []
            vimgs.append(vimg[0])
            vimgs.append(img2)
            vimgs.append(img3)
            vimgs.append(vimg[1])

        vimg = vimgs.copy()
        vres, init_Y, edge1, edge2 = model(*vimg)

        loss, L_L1, L_SSIM, L_Grad = Loss_Fun(e, vres, *vimgs)

        loss_list.append(loss.item())
        # print(-4)
        loss.backward()
        optimizer.step()

    loss_save=args.loss_log
    file_save=open(loss_save,mode='a')
    # file_save.write('\n'+'step:'+str(e) + '  Lint: '+str(Lint) + ' Ltext: '+str(Ltext) + ' Lssim: '+str(Lssim))
    file_save.write('\n'+'step:'+str(e) + '  Lint: '+str(L_L1)    + '   L_SSIM: '+str(L_SSIM) + '   L_Grad: '+str(L_Grad))
    # file_save.write('\n'+'step:'+str(e)+'  MSE: '+str(L2_loss) + '  SSIM: '+str(SSIM_loss))
    file_save.close()
        
    scheduler.step()
    
    # print()
    # scheduler.step()
    Loss_list.append(np.mean(loss_list))

    state = {
        'model': model.state_dict(),
        'loss': Loss_list
    }
    model_name = str(e) + '.pth'
    torch.save(state, os.path.join(args.model_path, model_name))

    matplotlib.use('Agg')
    fig_train = plt.figure()
    plt.plot(Loss_list)
    plt.savefig(args.figure_path)
    plt.close('all')

