#from Python
import time
import csv
import os
import math
import numpy as np
import sys
from shutil import copyfile

#from Pytorch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.utils as torch_utils
from torch.optim.lr_scheduler import StepLR

#from this project
from data_loader import get_loader
import data_loader as dl
import VisionOP
import model
import param as p
import utils 
import pytorch_ssim

#local function
def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def norm(x):
    out = (x - 0.5) * 2
    return out.clamp(-1,1)

################ Hyper Parameters ################
# VERSION
version = 'temp'
subversion = '1_1'

# data Set
dataSetName = p.dataSetName
dataSetMode = p.dataSetMode
dataPath = p.dataPath

maxDataNum = p.maxDataNum #in fact, 4500
batchSize = p.batchSize

MaxCropWidth = p.MaxCropWidth
MinCropWidth = p.MinCropWidth
MaxCropHeight = p.MaxCropHeight
MinCropHeight = p.MinCropHeight

# model
NOF = p.NOF

# train
MaxEpoch = p.MaxEpoch
learningRate = p.learningRate

# save
numberSaveImage = p.numberSaveImage

############################################


############################################
############################################
print("")
print("          _____  ______ _______ _____ _   _ ________   __  ")
print("         |  __ \\|  ____|__   __|_   _| \\ | |  ____\\ \\ / / ")
print("         | |__) | |__     | |    | | |  \\| | |__   \\ V / ")
print("         |  _  /|  __|    | |    | | | . ` |  __|   > < ")
print("         | | \\ \\| |____   | |   _| |_| |\\  | |____ / . \\")
print("         |_|  \\_\\______|  |_|  |_____|_| \\_|______/_/ \\_\\ ")                       
print("")
print("Retinex model")
print("main Version : " + version)
print("sub Version : " + subversion)
print("")
############################################
############################################
def psnr(input, target):
    #print(torch.min(input))
    #print(torch.max(input))
    input = torch.abs(input - target).cuda()

    MSE = torch.mean(input * input)

    PSNR = 10 * math.log10(1 / MSE)

    return PSNR

# system setting
MODE = sys.argv[1]

dataSetMode = 'train'
dataPath = './data/training_dataset/'

data_loader = get_loader(dataPath,MaxCropWidth,MinCropWidth,MaxCropHeight,MinCropHeight,batchSize,dataSetName,dataSetMode)


if not os.path.exists('data/' + version):
    os.makedirs('./data/' + version)
if not os.path.exists('./data/' + version + '/model'):
    os.makedirs('./data/' + version + '/model')
if not os.path.exists('./data/' + version + '/eval'):
    os.makedirs('./data/' + version + '/eval')
if not os.path.exists('./data/' + version + '/log'):
    os.makedirs('./data/' + version + '/log')
    f = open('./data/'+version+'/log/loss.csv', 'w', encoding='utf-8')
    wr = csv.writer(f)
    wr.writerow(["Epoch","LR","NETLoss"])
    f.close() 


#init model
Retinex = model.LMSN()
#Adjust = model.Adjust_class()

rt_optimizer = torch.optim.Adam(Retinex.parameters(), lr=learningRate)
rt_scheduler = StepLR(rt_optimizer,step_size = 600, gamma = 0.3)
#ad_optimizer = AdamW(Adjust.parameters(), lr=learningRate, weight_decay=0.0001)

Retinex = nn.DataParallel(Retinex).cuda()  
#Adjust = nn.DataParallel(Adjust).cuda() 


#model load
startEpoch = 0
print("Load below models")

if (MODE != 'n'):
    checkpoint_rt = torch.load('./data/' + version + '/Retinex' + '.pkl')
#    checkpoint_ad = torch.load('./data/' + version + '/Adjust' + '.pkl')

    Retinex.load_state_dict(checkpoint_rt['model'])
#    Adjust.load_state_dict(checkpoint_ad['model'])

    startEpoch = checkpoint_rt['epoch']

    rt_optimizer.load_state_dict(checkpoint_rt['optim'])
#    ad_optimizer.load_state_dict(checkpoint_ad['optim'])

    print("All model loaded.")

#Retinex.train()
#Adjust.train()


# loss 
scale1_loss = nn.MSELoss()
scale2_loss = nn.MSELoss()
scale3_loss = nn.MSELoss()
laplace_loss2 = nn.L1Loss()
laplace_loss3 = nn.L1Loss()
scale1_color = nn.CosineSimilarity(dim= 1, eps = 1e-6)
scale2_color = nn.CosineSimilarity(dim= 1, eps = 1e-6)
scale3_color = nn.CosineSimilarity(dim= 1, eps = 1e-6)


a = time.perf_counter()
k = time.perf_counter()

for epoch in range(startEpoch, MaxEpoch):

    # ============= Train Retinex & Adjust module =============#
    finali = 0

    AvgScale1Loss = 0
    AvgScale2Loss = 0
    AvgScale3Loss = 0
    AvgColor1Loss = 0
    AvgColor2Loss = 0
    AvgColor3Loss = 0
    AvgLaplace2Loss = 0
    AvgLaplace3Loss = 0

    rt_scheduler.step(epoch)

    for i, (images) in enumerate(data_loader):

        b,c,h,w_ = images.size()
        w = int(w_/2)
        
        Input = to_var(images[:,:,:,0:int(w_/2)]).contiguous()
        with torch.no_grad():
            GT = to_var(images[:,:,:,int(w_/2):]).contiguous()

        Scale1,Scale2,Scale3,res2,res3 = Retinex(Input)

        gt_down2 = F.interpolate(GT,scale_factor = 0.5,mode='bilinear') #128
        gt_down4 = F.interpolate(gt_down2,scale_factor = 0.5,mode='bilinear') #64

        in_down2 = F.interpolate(Input,scale_factor = 0.5,mode='bilinear') #128
        in_down4 = F.interpolate(in_down2,scale_factor = 0.5,mode='bilinear') #64

        reup2 = F.interpolate(gt_down4,scale_factor = 2,mode='bilinear') #128
        reup3 = F.interpolate(gt_down2,scale_factor = 2,mode='bilinear') #256

        laplace2 = gt_down2 - reup2
        laplace3 = GT - reup3

        scale3loss = scale3_loss(Scale3,GT)
        scale2loss = scale2_loss(Scale2,gt_down2)
        scale1loss = scale1_loss(Scale1,gt_down4)
        scale1color = torch.mean(-1 * scale1_color(Scale1,gt_down4))
        scale2color = torch.mean(-1 * scale2_color(Scale2,gt_down2))
        scale3color = torch.mean(-1 * scale3_color(Scale3,GT))
        laplaceloss2 = laplace_loss2(res2,laplace2)
        laplaceloss3 = laplace_loss3(res3,laplace3)

        loss = 2 * scale1loss + scale2loss + scale3loss + 2 * scale1color + scale2color + scale3color + laplaceloss2 + laplaceloss3


        #output = norm(denorm(ad_ill) * denorm(ref))
        #adjust_loss = recon_loss(output,GT)

        Retinex.zero_grad()
        loss.backward()
        rt_optimizer.step()





        AvgScale1Loss = AvgScale1Loss + torch.Tensor.item(scale1loss.data)  
        AvgScale2Loss = AvgScale2Loss + torch.Tensor.item(scale2loss.data)            
        AvgScale3Loss = AvgScale3Loss + torch.Tensor.item(scale3loss.data) 
        AvgColor1Loss = AvgColor1Loss + torch.Tensor.item(scale1color.data)  
        AvgColor2Loss = AvgColor2Loss + torch.Tensor.item(scale2color.data)            
        AvgColor3Loss = AvgColor3Loss + torch.Tensor.item(scale3color.data) 
        AvgLaplace2Loss = AvgLaplace2Loss + torch.Tensor.item(laplaceloss2.data) 
        AvgLaplace3Loss = AvgLaplace3Loss + torch.Tensor.item(laplaceloss3.data) 
            
        finali = i + 1

        if (i + 1) % 1 == 0:
            olda = a
            a = time.perf_counter()

        
            print('E[%d/%d][%.2f%%] NET:' % (epoch, MaxEpoch, (i + 1) / (maxDataNum / batchSize / 100)),  end=" ")
            print('scale3 : %.6f color3 : %.6f laplace2 : %.6f laplace3 : %.6f lr: %.6f, time: %.2f sec ' % (AvgScale3Loss / finali, AvgColor3Loss / finali, AvgLaplace2Loss / finali, AvgLaplace3Loss / finali,learningRate,(a-olda)), end="\r")


    oldk = k
    k = time.perf_counter()      


    print('E[%d/%d] NET:'
            % (epoch, MaxEpoch),  end=" ")

    AvgScale1Loss = AvgScale1Loss/finali
    AvgScale2Loss = AvgScale2Loss/finali 
    AvgScale3Loss = AvgScale3Loss/finali
    AvgColor1Loss = AvgColor1Loss/finali
    AvgColor2Loss = AvgColor2Loss/finali 
    AvgColor3Loss = AvgColor3Loss/finali

    print('scale1: %.6f scale2: %.6f scale3 : %.6f color1: %.6f color2: %.6f color3 : %.6f lr: %.6f, time: %.2f sec ' % (AvgScale1Loss, AvgScale2Loss,AvgScale3Loss,AvgColor1Loss, AvgColor2Loss,AvgColor3Loss, learningRate,(k-oldk)), end="\n")

    #save_image(torch.cat((Input.data,Scale3.data,GT.data),3), './data/'+version+'/enhanced%d_1.bmp' % (epoch + 1))
    #save_image(torch.cat((in_down2.data,Scale2.data,gt_down2.data),3), './data/'+version+'/enhanced%d_2.bmp' % (epoch + 1))
    #save_image(torch.cat((in_down4.data,Scale1.data,gt_down4.data),3), './data/'+version+'/enhanced%d_3.bmp' % (epoch + 1))


    # Save loss log
    #f = open('./data/'+version+'/log/loss.csv', 'a', encoding='utf-8')
    #wr = csv.writer(f)

    #wr.writerow([epoch,learningRate
    #,AvgScale3Loss
    #,AvgColor3Loss])
    #f.close()    

    torch.save({'model': Retinex.state_dict(), 'optim': rt_optimizer.state_dict(), 'epoch': epoch + 1}, './data/model/Retinex.pkl')






