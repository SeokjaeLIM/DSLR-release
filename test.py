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
version = '2019-12-19(LPGnet-with-LRblock)'
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

torch.backends.cudnn.benchmark = True

# system setting
MODE = sys.argv[1]

dataSetMode = 'test'
dataPath = './data/test/'

data_loader = get_loader(dataPath,MaxCropWidth,MinCropWidth,MaxCropHeight,MinCropHeight,batchSize,dataSetName,dataSetMode)


#init model
Retinex = model.LMSN()
Retinex = nn.DataParallel(Retinex).cuda()  



#model load
startEpoch = 0
print("Load below models")

if (MODE != 'n'):
    checkpoint_rt = torch.load('./data/model/Retinex' + '.pkl')
    Retinex.load_state_dict(checkpoint_rt['model'])
    print("All model loaded.")

def psnr(input, target):
    #print(torch.min(input))
    #print(torch.max(input))
    input = torch.abs(input - target).cuda()

    MSE = torch.mean(input * input)

    PSNR = 10 * math.log10(1 / MSE)

    return PSNR


#a = time.perf_counter()

for epoch in range(0, 1):

    # ============= Train Retinex & Adjust module =============#
    finali = 0
    ssim = 0
    psnr2 = 0

    torch.set_grad_enabled(False)

#    rt_scheduler.step(epoch)
    #total_time = 0
    j=0
    avg_in = 0
    avg_out = 0
    for i, (images) in enumerate(data_loader):

        
        b,c,h,w_ = images.size()
        w = int(w_/2)
        if i == 0:
            total_time = 0
   
        with torch.no_grad():
            torch.cuda.synchronize()
            Input = to_var(images).contiguous()
            if i >= 0:
                a = time.perf_counter()

                Scale1,Scale2,Scale3,res2,res3 = Retinex(Input)

                olda = a
                a = time.perf_counter()

                total_time = total_time + a - olda


                print('%d/500, time: %.5f sec ' % ((j+1),total_time / (j+1)), end="\n")
                j=j+1
            else:
                Scale1,Scale2,Scale3,res2,res3 = Retinex(Input)


            save_image(Scale3.data, './data/result/%d.png' % (i + 1))


