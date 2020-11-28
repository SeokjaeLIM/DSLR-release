import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torchvision.datasets import ImageFolder
from PIL import Image
import time

class enhance_dataset(Dataset):
    def __init__(self,image_path,transform,mode,MaxCropWidth,MinCropWidth,MaxCropHeight,MinCropHeight):
        self.image_path = image_path
        self.transform = transform
        self.MaxCropWidth = MaxCropWidth
        self.MinCropWidth = MinCropWidth
        self.MaxCropHeight = MaxCropHeight
        self.MinCropHeight = MinCropHeight
        self.crop_width = 0
        self.crop_height = 0
        self.mode = mode
        self.num_data = []
        self.list = []
        self.Folder_list = []
        self.image_filenames = []
        self.Img_list = []
        self.Blur_image = []
        self.Sharp_image = []
        self.temp = []
        self.k = 0

        print('Start image processing')
        self.image_processing()
        print('Fineshed image processing')

        if self.mode == 'train':
            self.num_data = int(len(self.image_filenames) / 2)
            print('TrainSet loaded : %d images'%self.num_data)

        if self.mode == 'test':
            self.num_data = int(len(self.image_filenames))
            print('TestSet loaded : %d images'%self.num_data)
           

    def image_processing(self):
        self.list = os.listdir(self.image_path)
        self.NOF = len(self.list)
        

        for i in range(0,self.NOF):
            self.Folder_list = self.image_path + self.list[i]
            self.Img_list = os.listdir(self.Folder_list)
            for j in range(0,len(self.Img_list)):
                self.temp.append(self.list[i] + '/' + self.Img_list[j])

        self.temp = sorted(self.temp)
        self.image_filenames.extend(self.temp)
        self.temp = []

    def get_params(self,input1):

        self.w, self.h = input1.size

        self.crop_width = 1024#random.randint(self.MinCropWidth,self.MaxCropWidth)
        self.crop_height = 1024#random.randint(self.MinCropHeight, self.MaxCropHeight)
      
        i = random.randint(0, self.h - self.crop_height)
        j = random.randint(0, self.w - self.crop_width)

        return i,j
        
    def Random_Crop(self,input1,input2):

        self.i,self.j = self.get_params((input1))

        image1 = input1.crop((self.j, self.i, self.j + self.crop_width, self.i + self.crop_height))

        image2 = input2.crop((self.j, self.i, self.j + self.crop_width, self.i + self.crop_height))

        return image1,image2
    '''
    def Resize(self,input):

        return self.resize(input)
    '''
    def FLIP_LR(self,input1,input2):
        if random.random() > 0.5:
            input1 = input1.transpose(Image.FLIP_LEFT_RIGHT)
            input2 = input2.transpose(Image.FLIP_LEFT_RIGHT)

        return input1, input2

    def FLIP_UD(self,input1,input2):
        if random.random() > 0.5:
            input1 = input1.transpose(Image.FLIP_TOP_BOTTOM)
            input2 = input2.transpose(Image.FLIP_TOP_BOTTOM)

        return input1, input2

    def __getitem__(self, index):

        image = Image.new('RGB', (1024 * 2, 1024))

        if self.mode == 'train':

            self.Input_image = Image.open(os.path.join(self.image_path, self.image_filenames[index + self.num_data]))
            self.Sharp_image = Image.open(os.path.join(self.image_path, self.image_filenames[index]))

            self.w, self.h = self.Input_image.size

            self.w = int(self.w - self.w % 256)
            self.h = int(self.h - self.h % 256)

            self.Input_image = self.Input_image.resize((int(self.w),int(self.h)))
            self.Sharp_image = self.Sharp_image.resize((int(self.w),int(self.h)))

            self.Input_image,self.Sharp_image = self.Random_Crop(self.Input_image,self.Sharp_image)

            #self.Blur_image = self.Resize(self.Blur_image)
            #self.Sharp_image = self.Resize(self.Sharp_image)

            self.Input_image, self.Sharp_image = self.FLIP_LR(self.Input_image, self.Sharp_image)
            self.Input_image, self.Sharp_image = self.FLIP_UD(self.Input_image, self.Sharp_image)

            image.paste(self.Input_image,(0,0))
            image.paste(self.Sharp_image,(1024,0))

            return self.transform(image)
            
        else:

            self.Input_image = Image.open(os.path.join(self.image_path, self.image_filenames[index ]))

            self.w, self.h = self.Input_image.size

            self.w = int(self.w - self.w % 256)
            self.h = int(self.h - self.h % 256)

            self.Input_image = self.Input_image.resize((int(self.w),int(self.h)))

            
            image = Image.new('RGB', ((int(self.w),int(self.h))))

            image.paste(self.Input_image,(0,0))

            return self.transform(image)
                
    def __len__(self):
        return self.num_data

def get_loader(image_path,MaxCropWidth,MinCropWidth,MaxCropHeight,MinCropHeight,batch_size,dataset='5K_dataset', mode='train'):
                
    """Build and return data loader."""
    transform = transforms.ToTensor()


    dataset = enhance_dataset(image_path, transform, mode,MaxCropWidth,MinCropWidth,MaxCropHeight,MinCropHeight)

    shuffle = False
    if mode == 'train':
        shuffle = True
        
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=4)
    return data_loader

