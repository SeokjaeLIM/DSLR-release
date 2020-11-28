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


# train
MaxEpoch = p.MaxEpoch
learningRateNET = p.learningRate

# save
numberSaveImage = p.numberSaveImage

# model
NDF = p.NOF
NDF2 = p.NOF2


############################################



class resBlock(nn.Module):
    
    def __init__(self, channelDepth, windowSize=3):
        
        super(resBlock, self).__init__()
        padding = math.floor(windowSize/2)
        self.conv1 = nn.Conv2d(channelDepth, channelDepth, windowSize, 1, padding)
        self.conv2 = nn.Conv2d(channelDepth, channelDepth, windowSize, 1, padding)

        self.IN_conv = nn.InstanceNorm2d(channelDepth,track_running_stats=False, affine=False)
                     
    def forward(self, x):     
        res = x
        x = F.relu(self.IN_conv(self.conv1(x)))
        x = self.IN_conv(self.conv2(x))
        x = F.relu(x+res)
        
        return x

class lrBLock_l3(nn.Module):

    def __init__(self, channelDepth, windowSize=3):

        super(lrBLock_l3, self).__init__()
        padding = math.floor(windowSize/2)

        self.res_l3 = resBlock(channelDepth,windowSize)
        self.res_l2 = resBlock(channelDepth,windowSize)
        self.res_l1 = resBlock(channelDepth,windowSize)

    def forward(self, x):

        x_down2 = F.interpolate(x,scale_factor = 0.5,mode='bilinear') #128
        x_down4 = F.interpolate(x_down2,scale_factor = 0.5,mode='bilinear') #64

        x_reup2 = F.interpolate(x_down4,scale_factor = 2,mode='bilinear') #128
        x_reup = F.interpolate(x_down2,scale_factor = 2,mode='bilinear') #256

        Laplace_2 = x_down2 - x_reup2
        Laplace_1 = x - x_reup

        Scale1 = self.res_l1(x_down4)
        Scale2 = self.res_l2(Laplace_2)
        Scale3 = self.res_l3(Laplace_1)

        output1 = Scale1
        output2 = F.interpolate(Scale1,scale_factor = 2,mode='bilinear') + Scale2
        output3 = F.interpolate(output2,scale_factor = 2,mode='bilinear') + Scale3

        return output3

class lrBLock_l2(nn.Module):

    def __init__(self, channelDepth, windowSize=3):

        super(lrBLock_l2, self).__init__()
        padding = math.floor(windowSize/2)

        self.res_l2 = resBlock(channelDepth,windowSize)
        self.res_l1 = resBlock(channelDepth,windowSize)

    def forward(self, x):

        x_down2 = F.interpolate(x,scale_factor = 0.5,mode='bilinear') #128

        x_reup = F.interpolate(x_down2,scale_factor = 2,mode='bilinear') #256

        Laplace_1 = x - x_reup

        Scale1 = self.res_l1(x_down2)
        Scale2 = self.res_l2(Laplace_1)

        output1 = Scale1
        output2 = F.interpolate(Scale1,scale_factor = 2,mode='bilinear') + Scale2

        return output2
        

class LMSB(nn.Module):
    def __init__(self):
        super(LMSB, self).__init__()    

        self.conv1_1 = nn.Conv2d(3, NDF * 1, 5, 2, 2)  # 3->32
        self.conv2_1 = nn.Conv2d(NDF * 1, NDF * 2, 5, 2, 2)  # 32->64
        self.conv3_1 = nn.Conv2d(NDF * 2, NDF * 4, 5, 2, 2)  # 64->128

        self.conv1_r1 = lrBLock_l3(NDF * 1,3)  # 64->64
        self.conv2_r1 = lrBLock_l3(NDF * 2,3)  # 128->128  
        self.conv3_r1 = lrBLock_l3(NDF * 4,3)  # 256->256
      

        self.Deconv1_1 = nn.ConvTranspose2d(NDF * 4, NDF * 2, 4, 2, 1)  # 128->64
        self.Deconv2_1 = nn.ConvTranspose2d(NDF * 2, NDF * 1, 4, 2, 1)  # 64->32
        self.Deconv3_1 = nn.ConvTranspose2d(NDF * 1, 3, 4, 2, 1)  # 64->3

        self.Deconv3_r1 = lrBLock_l3(NDF * 1,3)  # 64->64
        self.Deconv2_r1 = lrBLock_l3(NDF * 2,3)  # 128->128
        self.Deconv1_r1 = lrBLock_l3(NDF * 4,3)  # 256->256
      

        self.IN_conv1 = nn.InstanceNorm2d(NDF * 1,track_running_stats=False, affine=False)
        self.IN_conv2 = nn.InstanceNorm2d(NDF * 2,track_running_stats=False, affine=False)
        self.IN_conv3 = nn.InstanceNorm2d(NDF * 4,track_running_stats=False, affine=False)

    def forward(self, x):

        input = x
        x = F.relu(self.IN_conv1(self.conv1_1(x)))
        x1 = self.conv1_r1(x)

        x = F.relu(self.IN_conv2(self.conv2_1(x1)))
        x2 = self.conv2_r1(x)

        x = F.relu(self.IN_conv1(self.conv3_1(x2)))
        x = self.conv3_r1(x)

        x = self.Deconv1_r1(x)
        x = F.relu(self.IN_conv3(self.Deconv1_1(x)))

        x = self.Deconv2_r1(x+x2)
        x = F.relu(self.IN_conv3(self.Deconv2_1(x)))

        x = self.Deconv3_r1(x+x1)
        x = self.Deconv3_1(x) + input

        return x

class LMSB_2(nn.Module):
    def __init__(self):
        super(LMSB_2, self).__init__()    

        self.conv1_1 = nn.Conv2d(3, NDF2 * 1, 5, 2, 2)  # 3->32
        self.conv2_1 = nn.Conv2d(NDF2 * 1, NDF2 * 2, 5, 2, 2)  # 32->64
        self.conv3_1 = nn.Conv2d(NDF2 * 2, NDF2 * 4, 5, 2, 2)  # 64->128

        self.conv1_r1 = lrBLock_l2(NDF2 * 1,3)  # 64->64
        self.conv2_r1 = lrBLock_l2(NDF2 * 2,3)  # 128->128
        self.conv3_r1 = lrBLock_l2(NDF2 * 4,3)  # 256->256
        
        self.Deconv1_1 = nn.ConvTranspose2d(NDF2 * 4, NDF2 * 2, 4, 2, 1)  # 128->64
        self.Deconv2_1 = nn.ConvTranspose2d(NDF2 * 2, NDF2 * 1, 4, 2, 1)  # 64->32
        self.Deconv3_1 = nn.ConvTranspose2d(NDF2 * 1, 3, 4, 2, 1)  # 64->3

        self.Deconv3_r1 = lrBLock_l2(NDF2 * 1,3)  # 64->64
        self.Deconv2_r1 = lrBLock_l2(NDF2 * 2,3)  # 128->128
        self.Deconv1_r1 = lrBLock_l2(NDF2 * 4,3)  # 256->256
        

        self.IN_conv1 = nn.InstanceNorm2d(NDF2 * 1,track_running_stats=False, affine=False)
        self.IN_conv2 = nn.InstanceNorm2d(NDF2 * 2,track_running_stats=False, affine=False)
        self.IN_conv3 = nn.InstanceNorm2d(NDF2 * 4,track_running_stats=False, affine=False)

    def forward(self, x):

        input = x
        x = F.relu(self.IN_conv1(self.conv1_1(x)))
        x1 = self.conv1_r1(x)

        x = F.relu(self.IN_conv2(self.conv2_1(x1)))
        x2 = self.conv2_r1(x)

        x = F.relu(self.IN_conv1(self.conv3_1(x2)))
        x = self.conv3_r1(x)

        x = self.Deconv1_r1(x)
        x = F.relu(self.IN_conv3(self.Deconv1_1(x)))

        x = self.Deconv2_r1(x+x2)
        x = F.relu(self.IN_conv3(self.Deconv2_1(x)))

        x = self.Deconv3_r1(x+x1)
        x = self.Deconv3_1(x) + input

        return x


class LMSN(nn.Module):
    def __init__(self):
        super(LMSN, self).__init__()    

        self.Stage1 = LMSB()
        self.Stage2 = LMSB_2()
        self.Stage3 = LMSB_2()

    def forward(self, x):

        x_down2 = F.interpolate(x,scale_factor = 0.5,mode='bilinear') #128
        x_down4 = F.interpolate(x_down2,scale_factor = 0.5,mode='bilinear') #64

        x_reup2 = F.interpolate(x_down4,scale_factor = 2,mode='bilinear') #128
        x_reup = F.interpolate(x_down2,scale_factor = 2,mode='bilinear') #256

        Laplace_2 = x_down2 - x_reup2
        Laplace_1 = x - x_reup

        Scale1 = self.Stage1(x_down4)
        Scale2 = self.Stage2(Laplace_2)
        Scale3 = self.Stage3(Laplace_1)

        output1 = Scale1
        output2 = F.interpolate(Scale1,scale_factor = 2,mode='bilinear') + Scale2
        output3 = F.interpolate(output2,scale_factor = 2,mode='bilinear') + Scale3


        return output1,output2,output3,Scale2,Scale3


class Retinex_class(nn.Module):
    def __init__(self):
        super(Retinex_class, self).__init__()

        # Encoder module
        self.conv1 = nn.Conv2d(2, NDF * 1, 5, 2, 2)  # 3->64
        self.conv2 = nn.Conv2d(NDF * 1, NDF * 2, 5, 2, 2)  # 64->128
        self.conv3 = nn.Conv2d(NDF * 2, NDF * 4, 5, 2, 2)  # 128->256


        self.conv1_r1 = resBlock(NDF * 1,3)  # 64->64
        self.conv2_r1 = resBlock(NDF * 2,5)  # 128->128
        self.conv3_r1 = resBlock(NDF * 4,3)  # 256->256


        self.IN_conv1 = nn.InstanceNorm2d(NDF * 1,track_running_stats=False, affine=False)
        self.IN_conv2 = nn.InstanceNorm2d(NDF * 2,track_running_stats=False, affine=False)
        self.IN_conv3 = nn.InstanceNorm2d(NDF * 4,track_running_stats=False, affine=False)


        # Refinement module
        self.res_r1 = resBlock(NDF * 4,3) #512->512
        self.res_r2 = resBlock(NDF * 4,3) #512->512
        self.res_r3 = resBlock(NDF * 4,3) #512->512
        self.res_r4 = resBlock(NDF * 4,3) #512->512


        # Decomposition module
        self.Deconv1 = nn.ConvTranspose2d(NDF * 4, NDF * 2, 4, 2, 1)  # 256->128
        self.Deconv2 = nn.ConvTranspose2d(NDF * 2, NDF * 1, 4, 2, 1)  # 128->64
        self.Deconv3 = nn.ConvTranspose2d(NDF * 1, 1, 4, 2, 1)  # 64->3


        self.Deconv1_r1 = resBlock(NDF * 4,3)  # 256->256
        self.Deconv2_r1 = resBlock(NDF * 2,5)  # 128->128
        self.Deconv3_r1 = resBlock(NDF * 1,3)  # 64->64

        self.IN_deconv1 = nn.InstanceNorm2d(NDF * 4,track_running_stats=False, affine=False)
        self.IN_deconv2 = nn.InstanceNorm2d(NDF * 2,track_running_stats=False, affine=False)

    def forward(self, x,kernel):

        x = torch.cat((x,kernel),1)
        # Encoder module
        x = F.relu(self.IN_conv1(self.conv1(x)))
        x = self.conv1_r1(x)

        x = F.relu(self.IN_conv2(self.conv2(x)))
        x = self.conv2_r1(x)

        x = F.relu(self.IN_conv3(self.conv3(x)))
        x = self.conv3_r1(x)


        x = self.res_r1(x)
        x = self.res_r2(x)
        x = self.res_r3(x)
        x = self.res_r4(x)


        # decomposition
        x = self.Deconv1_r1(x)
        x = F.relu(self.IN_deconv1(self.Deconv1(x)))

        x = self.Deconv2_r1(x)
        x = F.relu(self.IN_deconv1(self.Deconv2(x)))

        x = self.Deconv3_r1(x)

        x = kernel * F.relu(self.Deconv3(x))


        return x
