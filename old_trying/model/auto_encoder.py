# original
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import pdb 
import os

def double_conv(in_channels, out_channels):
  return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding=1),
    nn.BatchNorm2d(out_channels),
    nn.ReLU(inplace=True),
    # nn.Dropout(p=0.5), ## regualarisation..using high dropout rate of 0.9...lets see for few moments...
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
    self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride = 2)
    self.conv = double_conv(in_channels, out_channels)
  def forward(self,x1,x2): 
    x1 = self.up(x1)
    # input is CHW 
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    x = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2]) 
    return self.conv(x)


def outconv(in_channels, out_channels):
  return nn.Conv2d(in_channels, out_channels, kernel_size=1)
 
class reconst_auto_encoder(nn.Module):
  def __init__(self, n_ip_chs=19, n_out_chs=19, base_chs=16):
    super().__init__()
    self.n_ip_chs = n_ip_chs
    self.n_out_chs = n_out_chs
    self.base_chs = base_chs    
    
    ## encoder defining   
    self.encoder = nn.Sequential(
      double_conv(self.n_ip_chs, self.base_chs),
      down(self.base_chs,self.base_chs*2),
      down(self.base_chs*2,self.base_chs*4),
      down(self.base_chs*4,self.base_chs*8), 
    )
        
    ## decoder defining
    self.decoder = nn.Sequential(
      up(self.base_chs*8,self.base_chs*4),
      up(self.base_chs*4,self.base_chs*2),
      up(self.base_chs*2,self.base_chs),
      outconv(self.base_chs,self.n_out_chs)
    )
  
  def forward(self, x):  
      
    x1 = self.encoder[0](x)
    x2 = self.encoder[1](x1)
    x3 = self.encoder[2](x2)
    x4 = self.encoder[3](x3) 
    
    x = self.decoder[0](x4,x3)
    x = self.decoder[1](x,x2)
    x = self.decoder[2](x,x1)
    logits = self.decoder[3](x)
    return logits
  
  
class denoise_auto_encoder(nn.Module):
  def __init__(self, n_ip_chs=128, n_out_chs=128, base_chs=128):
    super().__init__()
    self.n_ip_chs = n_ip_chs
    self.n_out_chs = n_out_chs
    self.base_chs = base_chs    
    
    ## encoder defining   
    self.encoder = nn.Sequential(
      double_conv(self.n_ip_chs, self.base_chs),
      down(self.base_chs,self.base_chs*2),
      down(self.base_chs*2,self.base_chs*4),
      )
      
    ## decoder defining
    self.decoder = nn.Sequential(
      up(self.base_chs*4,self.base_chs*2),
      up(self.base_chs*2,self.base_chs)
    )
      
  def forward(self, x):   
    
    x1 = self.encoder[0](x)
    x2 = self.encoder[1](x1)
    x3 = self.encoder[2](x2)
    
    # pdb.set_trace()
    x = self.decoder[0](x3,x2)
    logits = self.decoder[1](x,x1)  
    return logits



# if __name__ == '__main__': 
#   os.environ['CUDA_VISIBLE_DEVICES'] = '3'

#   print('*********************Model Summary***********************')
#   # model=  reconst_auto_encoder()
#   model=  denoise_auto_encoder()
#   if torch.cuda.is_available(): 
#     model = model.cuda()

#   print(summary(model, (128, 64, 64))) 
  # print(summary(model, (19, 512, 512))) 
