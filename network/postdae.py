from torchsummary import summary 
import torch 
import torch.nn as nn 
from torch.nn import functional as F  
import os 

def convrelu(in_channels, out_channels, kernel, stride ):
  return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel, stride=stride),
    nn.ReLU(inplace=True)
    # nn.Dropout2d(p=0.5) # new addition for the dropout case
  )

def upconvrelu(in_channels, out_channels, kernel, stride ):
  return nn.Sequential(
    nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=stride),
    nn.ReLU(inplace=True)
    # nn.Dropout2d(p=0.5) # new addition for the dropout case
  )

class Postdae(nn.Module): 
    def __init__(self, n_ip_chs, n_op_chs):
        super().__init__()
        
        self.layer1 = nn.Sequential(
            convrelu(n_ip_chs, 16, kernel=(3,3), stride=(2,2)),
            convrelu(16, 16, kernel=(3,3), stride=(1,1))
        )    
        self.layer2 = nn.Sequential(
            convrelu(16, 32, kernel=(3,3), stride=(2,2)),
            convrelu(32, 32, kernel=(3,3), stride=(1,1))
        )
        self.layer3 = nn.Sequential(
            convrelu(32, 32, kernel=(3,3), stride=(2,2)),
            convrelu(32, 32, kernel=(3,3), stride=(1,1))
        )
        self.layer4  = nn.Sequential(
            convrelu(32, 32, kernel=(3,3), stride=(2,2)),
            convrelu(32, 32, kernel=(3,3), stride=(1,1))
        )
        self.layer5 = convrelu(32, 32, kernel=(3,3), stride=(2,2)) 
        self.layer6 = nn.Linear(32, 512)
        self.layer7 = nn.Sequential(
            nn.Linear(512, 1024), 
            nn.ReLU()
        )
        self.layer8 = nn.Sequential(
            upconvrelu(1024, 16, kernel=(3,3), stride=(1,1)), # have to change to (2,2) else mostly won't work
            convrelu(16, 16, kernel=(3,3), stride=(1,1)) 
        )
        self.layer9 = nn.Sequential(
            upconvrelu(16, 16, kernel=(3,3), stride=(1,1)), # have to change to (2,2) else mostly won't work
            convrelu(16, 16, kernel=(3,3), stride=(1,1)) 
        )
        self.layer10 = nn.Sequential(
            upconvrelu(16, 16, kernel=(3,3), stride=(1,1)), # have to change to (2,2) else mostly won't work
            convrelu(16, 16, kernel=(3,3), stride=(1,1)) 
        )
        self.layer11 = nn.Sequential(
            upconvrelu(16, 16, kernel=(3,3), stride=(1,1)), # have to change to (2,2) else mostly won't work
            convrelu(16, 16, kernel=(3,3), stride=(1,1)) 
        )
        self.layer12 = nn.Sequential(
            upconvrelu(16, 16, kernel=(3,3), stride=(1,1)), # have to change to (2,2) else mostly won't work
            convrelu(16, n_op_chs, kernel=(3,3), stride=(1,1)) 
        )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x) 
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x) 
        
        return x 
    
if __name__ == '__main__': 
  os.environ['CUDA_VISIBLE_DEVICES'] = '6'

  print('*********************Model Summary***********************')
  model = Postdae(3, 19)
  if torch.cuda.is_available(): 
    model = model.cuda() 

print(summary(model, (3, 1024, 1024)))   
        
        