# original
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import os

def double_conv(in_channels, out_channels):
  return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding=1),
    nn.BatchNorm2d(out_channels),
    nn.ReLU(inplace=True),
    # nn.Dropout(p=0.9), ## regualarisation..using high dropout rate of 0.9...lets see for few moments...
    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
    nn.BatchNorm2d(out_channels),
    nn.ReLU(inplace=True),
    # nn.Dropout(p=0.9) ## dual dropout 
  )

def down(in_channels, out_channels):
  ## downsampling with maxpool then double conv
  return nn.Sequential(
    nn.MaxPool2d(2),
    double_conv(in_channels, out_channels)
  )

class up(nn.Module):
  ## upsampling then double conv 
  def __init__(self, in_channels, out_channels):
    super(up, self).__init__()
    self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride = 2)
    self.conv = double_conv(in_channels, out_channels)
  def forward(self,x1,x2): 
    x1 = self.up(x1)
    # input is CHW 
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])
    # if you have padding issues, see
    # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
    # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
    x = torch.cat([x2, x1], dim=1)
    return self.conv(x)


def outconv(in_channels, out_channels):
  return nn.Conv2d(in_channels, out_channels, kernel_size=1)
 

class UNet_mod(nn.Module):
  def __init__(self, n_channels, n_class, small=False):
    super(UNet_mod, self).__init__()
    self.n_channels = n_channels
    self.n_class = n_class
    self.small = small 

    self.inc = double_conv(self.n_channels, 64)
    self.down1 = down(64,128)
    self.down2 = down(128,256)
    self.down3 = down(256,512)
    self.down4 = down(512, 1024)
    
    self.up1 = up(1024,512)  
    self.up2 = up(512,256)
    self.up3 = up(256,128)
    self.up4 = up(128,64)
    self.out = outconv(64,self.n_class)
  
  def forward(self, x):
    if self.small: 
      x1 = self.inc(x)
      x2 = self.down1(x1)
      x3 = self.down2(x2)
      x4 = self.down3(x3)

      x = self.up2(x4,x3)
      x = self.up3(x,x2)
      x = self.up4(x,x1)
      logits = self.out(x)

    else:
      x1 = self.inc(x)
      x2 = self.down1(x1)
      x3 = self.down2(x2)
      x4 = self.down3(x3)
      x5 = self.down4(x4)
      
      x = self.up1(x5,x4)
      x = self.up2(x,x3)
      x = self.up3(x,x2)
      x = self.up4(x,x1)
      logits = self.out(x)

    return logits

if __name__ == '__main__': 
  os.environ['CUDA_VISIBLE_DEVICES'] = '3'

  print('*********************Model Summary***********************')
  model = UNet_mod(n_channels=19, n_class = 19) 
  if torch.cuda.is_available(): 
    model = model.cuda()

  print(summary(model, (19, 512, 512))) 
# original 
