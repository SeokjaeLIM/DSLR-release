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


version = '5.4-Freq'
eps = 0.000001

def Laplacian(input,ksize):

    if ksize==1:
        kernel = torch.tensor(  [0,1,0],
                                [1,-4,1],
                                [0,1,0])
    else:
        kernel = torch.ones(ksize,ksize)
        kernel[int(ksize/2),int(ksize/2)] = 1 - ksize**2

    kernel = Variable(kernel.view(1,1,1,ksize,ksize))
    output = F.conv3d(input.view(input.size()[0],1,3,input.size()[2],input.size()[3]),kernel,padding = [0, int(ksize/2), int(ksize/2)]).view(input.size()[0],-1,input.size()[2],input.size()[3])
    
    return output


def Gaussian(input,ksize,sigma):

    ax = np.arange(-ksize // 2 + 1., ksize // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / np.sum(kernel)

    kernel = Variable((torch.from_numpy(kernel).float()).cuda()).view(1,1,1,ksize,ksize)
    output = F.conv3d(input.view(input.size()[0],1,3,input.size()[2],input.size()[3]),kernel,padding = [0, int(ksize/2), int(ksize/2)]).view(input.size()[0],-1,input.size()[2],input.size()[3])
    
    return output

def Gaussiankernel(b,c,h,w,sigma):

    ax = np.arange(-w // 2 + 1., w // 2 + 1.)
    ay = np.arange(-h // 2 + 1., h // 2 + 1.)
    xx, yy = np.meshgrid(ax, ay)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / np.max(kernel)
    kernel = np.fft.fftshift(kernel)

    kernel = Variable((torch.from_numpy(kernel).float()).cuda()).view(1,1,h,w)
    kernel = kernel.repeat(b,c,1,1)
    
    return kernel


def HLCombine(inputH,inputL,kernel):
    H_mag,H_pha = polarFFT(inputH)
    L_mag,L_pha = polarFFT(inputL)
    '''
    ax = np.arange(-inputH.size(2) // 2 + 1., inputH.size(3) // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / np.sum(kernel)
    
    kernel = Variable((torch.from_numpy(kernel).float()).cuda()).view(1,1,inputH.size(2),inputH.size(3))
    '''
    kernel = kernel.expand(inputH.size(0),inputH.size(1),inputH.size(2),inputH.size(3))
    #kernel = kernel / torch.max(kernel)
    
    H_mag = torch.cat((H_mag[:,:,int(inputH.size(2)/2):,:],H_mag[:,:,0:int(inputH.size(2)/2),:]),2)
    H_mag = torch.cat((H_mag[:,:,:,int(inputH.size(2)/2):],H_mag[:,:,:,0:int(inputH.size(2)/2)]),3)

    L_mag = torch.cat((L_mag[:,:,int(inputH.size(2)/2):,:],L_mag[:,:,0:int(inputH.size(2)/2),:]),2)
    L_mag = torch.cat((L_mag[:,:,:,int(inputH.size(2)/2):],L_mag[:,:,:,0:int(inputH.size(2)/2)]),3)

    H_mag = H_mag * (1-kernel)
    L_mag = L_mag * kernel

    H_mag = torch.cat((H_mag[:,:,int(inputH.size(2)/2):,:],H_mag[:,:,0:int(inputH.size(2)/2),:]),2)
    H_mag = torch.cat((H_mag[:,:,:,int(inputH.size(2)/2):],H_mag[:,:,:,0:int(inputH.size(2)/2)]),3)

    L_mag = torch.cat((L_mag[:,:,int(inputH.size(2)/2):,:],L_mag[:,:,0:int(inputH.size(2)/2),:]),2)
    L_mag = torch.cat((L_mag[:,:,:,int(inputH.size(2)/2):],L_mag[:,:,:,0:int(inputH.size(2)/2)]),3)

    H_f_r = H_mag * torch.cos(H_pha)
    H_f_i = H_mag * torch.sin(H_pha)
    H_f = torch.cat((H_f_r.view(H_f_r.size(0),H_f_r.size(1),H_f_r.size(2),H_f_r.size(3),1),H_f_i.view(H_f_i.size(0),H_f_i.size(1),H_f_i.size(2),H_f_i.size(3),1)),4)

    L_f_r = L_mag * torch.cos(L_pha)
    L_f_i = L_mag * torch.sin(L_pha)
    L_f = torch.cat((L_f_r.view(L_f_r.size(0),L_f_r.size(1),L_f_r.size(2),L_f_r.size(3),1),L_f_i.view(L_f_i.size(0),L_f_i.size(1),L_f_i.size(2),L_f_i.size(3),1)),4)

    return torch.irfft(H_f + L_f, 2, onesided=False)
    
    


def polarIFFT(x_mag, x_pha):
    #x_mag = torch.cat((x_mag[:,:,128:,:],x_mag[:,:,0:128,:]),2)
    #x_mag = torch.cat((x_mag[:,:,:,128:],x_mag[:,:,:,0:128]),3)

    x_f_r = x_mag * torch.cos(x_pha)
    x_f_i = x_mag * torch.sin(x_pha)
    x_f = torch.cat((x_f_r.view(x_f_r.size(0),x_f_r.size(1),x_f_r.size(2),x_f_r.size(3),1),x_f_i.view(x_f_r.size(0),x_f_r.size(1),x_f_r.size(2),x_f_r.size(3),1)),4)
    
    output = torch.irfft(x_f,2,onesided=False)
    return output




def polarFFT(input):
    x_f = torch.rfft(input,2,onesided=False)
    x_f_r = x_f[:,:,:,:,0].contiguous()
    x_f_i = x_f[:,:,:,:,1].contiguous()

    x_f_r[x_f_r==0] = eps
    x_mag = torch.sqrt(x_f_r*x_f_r + x_f_i*x_f_i)

    #x_mag = (x_mag - torch.min(x_mag))/(torch.max(x_mag) - torch.min(x_mag))
    
    x_pha = torch.atan2(x_f_i,x_f_r)
 
    

    return x_mag,x_pha
    

def HPFinFreq(input,sigma,freqReturn=False):
    ax = np.arange(-input.size(2) // 2 + 1., input.size(3) // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / np.sum(kernel)

    kernel = Variable((torch.from_numpy(kernel).float()).cuda()).view(1,1,input.size(2),input.size(3))
    kernel = kernel.expand(input.size(0),input.size(1),input.size(2),input.size(3))
    kernel = kernel / torch.max(kernel)
    x_f = torch.rfft(input,2,onesided=False)
    x_f_r = x_f[:,:,:,:,0].contiguous()
    x_f_i = x_f[:,:,:,:,1].contiguous()
    x_mag = torch.sqrt(x_f_r*x_f_r + x_f_i*x_f_i)
    x_pha = torch.atan2(x_f_i,x_f_r)

    x_mag = torch.cat((x_mag[:,:,128:,:],x_mag[:,:,0:128,:]),2)
    x_mag = torch.cat((x_mag[:,:,:,128:],x_mag[:,:,:,0:128]),3)

    #x_pha = torch.cat((x_pha[:,:,128:,:],x_pha[:,:,0:128,:]),2)
    #x_pha = torch.cat((x_pha[:,:,:,128:],x_pha[:,:,:,0:128]),3)
    
    x_mag = x_mag * (1-kernel)
    #x_pha = x_pha * (1-kernel)

    x_mag = torch.cat((x_mag[:,:,128:,:],x_mag[:,:,0:128,:]),2)
    x_mag = torch.cat((x_mag[:,:,:,128:],x_mag[:,:,:,0:128]),3)

    #x_pha = torch.cat((x_pha[:,:,128:,:],x_pha[:,:,0:128,:]),2)
    #x_pha = torch.cat((x_pha[:,:,:,128:],x_pha[:,:,:,0:128]),3)
    
    x_f_r = x_mag * torch.cos(x_pha)
    x_f_i = x_mag * torch.sin(x_pha)
    x_f = torch.cat((x_f_r.view(x_f_r.size(0),x_f_r.size(1),x_f_r.size(2),x_f_r.size(3),1),x_f_i.view(x_f_r.size(0),x_f_r.size(1),x_f_r.size(2),x_f_r.size(3),1)),4)

    output = x_mag,x_pha

    if(freqReturn == False):
        output = torch.irfft(x_f,2,onesided=False)
    
    return output

def LPFinFreq(input,sigma,freqReturn=False):
    ax = np.arange(-input.size(2) // 2 + 1., input.size(3) // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / np.sum(kernel)

    kernel = Variable((torch.from_numpy(kernel).float()).cuda()).view(1,1,input.size(2),input.size(3))
    kernel = kernel.expand(input.size(0),input.size(1),input.size(2),input.size(3))
    kernel = kernel / torch.max(kernel)
    x_f = torch.rfft(input,2,onesided=False)
    x_f_r = x_f[:,:,:,:,0].contiguous()
    x_f_i = x_f[:,:,:,:,1].contiguous()
    x_mag = torch.sqrt(x_f_r*x_f_r + x_f_i*x_f_i)
    x_pha = torch.atan2(x_f_i,x_f_r)

    x_mag = torch.cat((x_mag[:,:,128:,:],x_mag[:,:,0:128,:]),2)
    x_mag = torch.cat((x_mag[:,:,:,128:],x_mag[:,:,:,0:128]),3)
    
    x_mag = x_mag * kernel

    x_mag = torch.cat((x_mag[:,:,128:,:],x_mag[:,:,0:128,:]),2)
    x_mag = torch.cat((x_mag[:,:,:,128:],x_mag[:,:,:,0:128]),3)
    
    x_f_r = x_mag * torch.cos(x_pha)
    x_f_i = x_mag * torch.sin(x_pha)
    x_f = torch.cat((x_f_r.view(x_f_r.size(0),x_f_r.size(1),x_f_r.size(2),x_f_r.size(3),1),x_f_i.view(x_f_r.size(0),x_f_r.size(1),x_f_r.size(2),x_f_r.size(3),1)),4)

    output = x_mag,x_pha

    if(freqReturn == False):
        output = torch.irfft(x_f,2,onesided=False)
    
    return output

def InverseLPFinFreq(input,sigma):
    ax = np.arange(-input.size(2) // 2 + 1., input.size(3) // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / np.sum(kernel)

    kernel = Variable((torch.from_numpy(kernel).float()).cuda()).view(1,1,input.size(2),input.size(3))
    kernel = kernel.expand(input.size(0),input.size(1),input.size(2),input.size(3))
    kernel = 1/(kernel / torch.max(kernel))
    x_f = torch.rfft(input,2,onesided=False)
    x_f_r = x_f[:,:,:,:,0].contiguous()
    x_f_i = x_f[:,:,:,:,1].contiguous()
    x_mag = torch.sqrt(x_f_r*x_f_r + x_f_i*x_f_i)
    x_pha = torch.atan2(x_f_i,x_f_r)

    x_mag = torch.cat((x_mag[:,:,128:,:],x_mag[:,:,0:128,:]),2)
    x_mag = torch.cat((x_mag[:,:,:,128:],x_mag[:,:,:,0:128]),3)
    
    x_mag = x_mag * kernel

    x_mag = torch.cat((x_mag[:,:,128:,:],x_mag[:,:,0:128,:]),2)
    x_mag = torch.cat((x_mag[:,:,:,128:],x_mag[:,:,:,0:128]),3)

    x_f_r = x_mag * torch.cos(x_pha)
    x_f_i = x_mag * torch.sin(x_pha)
    x_f = torch.cat((x_f_r.view(x_f_r.size(0),x_f_r.size(1),x_f_r.size(2),x_f_r.size(3),1),x_f_i.view(x_f_r.size(0),x_f_r.size(1),x_f_r.size(2),x_f_r.size(3),1)),4)
    
    output = torch.irfft(x_f,2,onesided=False)
    
    return output

def Nonzero(input): # replace nonzero elements to 1s

    out = torch.ones(input.size())
    out = eps*out.cuda()
    input = (torch.abs(input)).cuda()
    out = Min(input, out)
    out = out / eps

    return out

def Max(A,B): #3 ch input
    
    return torch.max(torch.cat((A.unsqueeze(-1),B.unsqueeze(-1)),-1),-1)[0]

def Min(A,B):

    return torch.min(torch.cat((A.unsqueeze(-1),B.unsqueeze(-1)),-1),-1)[0]

def FilterByValue(input, min, max):  # min<=input<max
    
    th1 = Max((input - min),torch.zeros(input.size()).cuda()) / ((input - min) + eps)
    
    th1s = 1 - Nonzero(input - min)
    th1 = th1 + th1s
    
    th2 = Max(-(input - max),torch.zeros(input.size()).cuda()) / (-(input - max) + eps)
    return input*th1*th2
    

def RGB2HSI(input): #c0:Hue(0~1) c1:Sat(0~1) c2:Value(0-1)
    input = (1-Nonzero(input))*eps + input
    
    R = torch.squeeze(input[:,0,:,:])
    G = torch.squeeze(input[:,1,:,:])
    B = torch.squeeze(input[:,2,:,:])

    Allsame = (1 - Nonzero(R - G)) * (1 - Nonzero(R - B))
    BGsame = (1 - Nonzero(G - B))


    nBgtG = (1 - Allsame) * Nonzero(Max((B - G),torch.zeros(G.size()).cuda())) + BGsame
    nGgtB = (1 - Allsame) * Nonzero(Max((G - B),torch.zeros(G.size()).cuda()))

    
    BgtG = B * nBgtG  # element of B that <G was replaced by 0
    GltB = G * nBgtG
    
    GgtB = G * nGgtB # element of G that <B was replaced by 0
    BltG = B * nGgtB

    H_GgtB = torch.acos((R-GgtB+R-BltG)/(2*torch.sqrt(eps + (R-GgtB)*(R-GgtB)+(R-BltG)*(GgtB-BltG)))) * nGgtB
    H_BgtG = (2*math.pi - torch.acos(torch.clamp((R-GltB+R-BgtG)/(2*torch.sqrt(eps + (R-GltB)*(R-GltB)+(R-BgtG)*(GltB-BgtG))),-1+eps,1-eps))) * nBgtG
    
    H = (H_GgtB + H_BgtG) / (2*math.pi)
    I = (R+G+B)/3
    S = 1-3*torch.min(input,1)[0]/(R+G+B)
    

    output = torch.cat((H.view(input.size()[0],1,input.size()[2],input.size()[3]),
                        S.view(input.size()[0],1,input.size()[2],input.size()[3]),
                        I.view(input.size()[0],1,input.size()[2],input.size()[3])),1)
    
    return output

def HSI2RGB(input):     

    H = torch.squeeze(input[:,0,:,:]) * 6
    S = torch.squeeze(input[:,1,:,:])
    I = torch.squeeze(input[:,2,:,:])

    
    '''
    Z = 1 - torch.abs(torch.fmod(H,2) - 1)
    C = 3*I*S / (I+Z)
    X = C*Z
    '''
    
    H02 = FilterByValue(H,0,2) / H
    H24 = FilterByValue(H,2,4) / H
    H46 = FilterByValue(H,4,6.01) / H

    H = H/6

    b = (H02 * ((1 - S) / 3) +
        H24 * ((1 - ((1 - S) / 3) - (1+S*torch.cos((H-1/3)*2*math.pi)/(torch.cos((1/6 - (H-1/3))*2*math.pi)+eps))/3)) +
        H46 * ((1+S*torch.cos((H-2/3)*2*math.pi)/(torch.cos((1/6 - (H-2/3))*2*math.pi)+eps))/3))
    
    r = (H24 * ((1 - S) / 3) +
        H46 * ((1 - ((1 - S) / 3) - (1+S*torch.cos((H-2/3)*2*math.pi)/(torch.cos((1/6 - (H-2/3))*2*math.pi)+eps))/3)) +
        H02 * ((1+S*torch.cos(H*2*math.pi)/(torch.cos((1/6 - H)*2*math.pi)+eps))/3))

    g = (H46 * ((1 - S) / 3) +
        H02 * ((1 - ((1 - S) / 3) - (1+S*torch.cos(H*2*math.pi)/(torch.cos((1/6 - H)*2*math.pi)+eps))/3)) +
        H24 * ( ( 1 + S*torch.cos((H-1/3)*2*math.pi) / (torch.cos((1/6 - (H-1/3))*2*math.pi)+eps))/3))

    
    R = 3*I*r
    G = 3*I*g
    B = 3*I*b

    output = torch.cat((R.view(input.size()[0],1,input.size()[2],input.size()[3]),
                        G.view(input.size()[0],1,input.size()[2],input.size()[3]),
                        B.view(input.size()[0],1,input.size()[2],input.size()[3])),1)
    
    return output
'''
A = torch.tensor([0.486275,0.458824,0.458824]).cuda().view(1,3,1,1)
print(A)
A = RGB2HSI(A)
print(A)
A = HSI2RGB(A)
print(A)
'''
