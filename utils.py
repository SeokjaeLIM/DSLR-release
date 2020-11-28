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
from torch.autograd import Variable,grad
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image

#from this project
import param as p
import VisionOP

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
maxDataNum = p.maxDataNum #in fact, 4206
batchSize = p.batchSize

MaxCropWidth = p.MaxCropWidth
MinCropWidth = p.MinCropWidth
MaxCropHeight = p.MaxCropHeight
MinCropHeight = p.MinCropHeight

NOF = p.NOF

# train
MaxEpoch = p.MaxEpoch
learningRateNET = p.learningRate

# save
numberSaveImage = p.numberSaveImage


############################################
class PSNR(nn.Module):

    def __init__(self):
        super(PSNR, self).__init__()

        self.MSE = 0

        # (N,C,H,W)

    def forward(self, input, target):
        input = torch.abs(input - target).cuda()

        self.MSE = torch.mean(input * input)

        PSNR = 10 * math.log10((255 * 255) / self.MSE)

        return PSNR

class Smooth_loss(nn.Module):
    def __init__(self,Smooth_weight=1):
        super(Smooth_loss,self).__init__()
        
        self.Smooth_weight = Smooth_weight

    def forward(self,x):
        b,c,h,w = x.size()

        x_h = F.pad(x,(0,0,1,1))
        h_tv = torch.mean(torch.pow((x_h[:,:,2:,:]-x_h[:,:,:h,:]),2))
        x_w = F.pad(x,(1,1,0,0))
        w_tv = torch.mean(torch.pow((x_w[:,:,:,2:]-x_w[:,:,:,:w]),2))
        #h_tv = torch.mean(torch.pow((x[:,:,1:,:]-x[:,:,:h-1,:]),2))
        #w_tv = torch.mean(torch.pow((x[:,:,:,1:]-x[:,:,:,:w-1]),2))
        self.loss = (h_tv + w_tv) / 2

        return self.loss

class Sparse_loss(nn.Module):
    def __init__(self,option=1):            #option 1: normal loss, option 2: weighting loss
        super(Sparse_loss,self).__init__()
        
        self.option = option

    def forward(self,x,input):
        b,c,h,w = x.size()

        x_h = F.pad(x,(0,0,1,1))
        x_w = F.pad(x,(1,1,0,0))
        if self.option == 1:
            h_tv = torch.abs(x_h[:,:,2:,:]-x_h[:,:,:h,:])
            w_tv = torch.abs(x_w[:,:,:,2:]-x_w[:,:,:,:w])
        else:
            input_h = F.pad(x,(0,0,1,1))
            input_w = F.pad(x,(1,1,0,0))  

            input_grad_h = torch.abs(input_h[:,:,2:,:]-input_h[:,:,:h,:])
            input_grad_w = torch.abs(input_w[:,:,:,2:]-input_w[:,:,:,:w])

            x_grad_h = torch.abs(x_h[:,:,2:,:]-x_h[:,:,:h,:])
            x_grad_w = torch.abs(x_w[:,:,:,2:]-x_w[:,:,:,:w])

            h_ = 1 / (255*input_grad_h + 0.0001) * x_grad_h
            w_ = 1 / (255*input_grad_w + 0.0001) * x_grad_w

            h_tv = torch.mean(h_)
            w_tv = torch.mean(w_)
        self.loss = h_tv + w_tv

        return self.loss        




def tv_loss(img):
    """
    Compute total variation loss.
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    b,c,h,w_ = img.size()
    w_variance = torch.sum(torch.pow(img[:,:,:,:-1] - img[:,:,:,1:], 2))/b
    h_variance = torch.sum(torch.pow(img[:,:,:-1,:] - img[:,:,1:,:], 2))/b
    loss = (h_variance + w_variance) / 2
    return loss