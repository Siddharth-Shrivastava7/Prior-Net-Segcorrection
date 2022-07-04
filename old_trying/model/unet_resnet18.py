import torch.nn as nn
import torchvision.models
import torch
from torchsummary import summary 
import os 
import torch.nn.functional as F

def convrelu(in_channels, out_channels, kernel, padding):
  return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
    nn.ReLU(inplace=True),
    # nn.Dropout2d(p=0.5) # new addition for the dropout case
  )

def adjustpad_cat(x1,x2): 
    # input is CHW 
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]  
    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])
    # if you have padding issues, see
    # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
    # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
    x = torch.cat([x2, x1], dim=1)
    return x 

class ResNet18UNet(nn.Module):
  def __init__(self, n_ip_channels, n_class):
    super().__init__()

    self.base_model = torchvision.models.resnet18(pretrained=True) 
    self.base_layers = list(self.base_model.children())
    # breakpoint() 

    # self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
    self.layer0 = nn.Sequential(
            nn.Conv2d(n_ip_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False), 
            nn.Sequential(*self.base_layers[1:3]) 
        )  ## for 19 channel input
    # breakpoint()  
    self.layer0_1x1 = convrelu(64, 64, 1, 0)
    self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
    self.layer1_1x1 = convrelu(64, 64, 1, 0)
    self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
    self.layer2_1x1 = convrelu(128, 128, 1, 0)
    self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
    self.layer3_1x1 = convrelu(256, 256, 1, 0)
    self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
    self.layer4_1x1 = convrelu(512, 512, 1, 0)

    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
    self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
    self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
    self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

    # self.conv_original_size0 = convrelu(3, 64, 3, 1)
    self.conv_original_size0 = convrelu(n_ip_channels, 64, 3, 1)
    self.conv_original_size1 = convrelu(64, 64, 3, 1)
    self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

    self.conv_last = nn.Conv2d(64, n_class, 1)

  def forward(self, input):
    x_original = self.conv_original_size0(input)
    x_original = self.conv_original_size1(x_original)

    layer0 = self.layer0(input)
    layer1 = self.layer1(layer0)
    layer2 = self.layer2(layer1)
    layer3 = self.layer3(layer2)
    layer4 = self.layer4(layer3)

    layer4 = self.layer4_1x1(layer4)
    x = self.upsample(layer4)
    layer3 = self.layer3_1x1(layer3)
    # x = torch.cat([x, layer3], dim=1) 
    x = adjustpad_cat(x, layer3)
    x = self.conv_up3(x)

    x = self.upsample(x)
    layer2 = self.layer2_1x1(layer2)
    # x = torch.cat([x, layer2], dim=1) 
    x = adjustpad_cat(x, layer2)
    x = self.conv_up2(x)

    x = self.upsample(x)
    layer1 = self.layer1_1x1(layer1)
    # x = torch.cat([x, layer1], dim=1) 
    x = adjustpad_cat(x, layer1)
    x = self.conv_up1(x)

    x = self.upsample(x)
    layer0 = self.layer0_1x1(layer0)
    # x = torch.cat([x, layer0], dim=1) 
    x = adjustpad_cat(x, layer0)
    x = self.conv_up0(x)

    x = self.upsample(x)
    # x = torch.cat([x, x_original], dim=1)
    x = adjustpad_cat(x, x_original)
    x = self.conv_original_size2(x)
    
    out = self.conv_last(x)
    return out

if __name__ == '__main__': 
  os.environ['CUDA_VISIBLE_DEVICES'] = '6'

  print('*********************Model Summary***********************')
  model = ResNet18UNet(3, 3)
  if torch.cuda.is_available(): 
    model = model.cuda() 

print(summary(model, (3, 512, 512)))   