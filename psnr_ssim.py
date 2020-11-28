import pytorch_ssim
import torch
from torch.autograd import Variable
import data_loader_test
import math

maxDataNum = 992

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)
def psnr2(input, target):
    #print(torch.min(input))
    #print(torch.max(input))
    input = torch.abs(input - target).cuda()

    MSE = torch.mean(input * input)

    PSNR = 10 * math.log10((255 * 255) / MSE)

    return PSNR
gt_dataset = data_loader_test.InpaintingDataset(root_dir='./demo/gt/')
#gt_dataset = data_loader_test.InpaintingDataset(root_dir='../FaceForensics/frame/RandomHoles/gt/')
gt_loader = torch.utils.data.DataLoader(gt_dataset,
                                             batch_size=1, shuffle=False)
out_dataset = data_loader_test.InpaintingDataset(root_dir='./demo/output/')
out_loader = torch.utils.data.DataLoader(out_dataset,
                                          batch_size=1, shuffle=False)
cnt=0
ssim=0
psnr=0
for i, (sample1,sample2) in enumerate(zip(gt_loader,out_loader)):
    gt=Variable(sample1['image']).cuda()
    out=Variable(sample2['image']).cuda()
    ssim+=pytorch_ssim.ssim(gt,out)
    psnr+=psnr2(denorm(gt)*255,denorm(out)*255)
    print(cnt/maxDataNum*100)
    cnt=cnt+1

ssim=ssim/cnt
psnr=psnr/cnt
print('ssim_average : %.4f'%ssim)
print('psnr_average : %.4f'%psnr)


'''
img1 = Variable(torch.rand(1, 1, 256, 256))
img2 = Variable(torch.rand(1, 1, 256, 256))

if torch.cuda.is_available():
    img1 = img1.cuda()
    img2 = img2.cuda()

print('pytorch_ssim: %.2f'%pytorch_ssim.ssim(img1, img2))

ssim_loss = pytorch_ssim.SSIM(window_size = 11)

print('ssimloss: %.2f'%ssim_loss(img1, img2))
'''
