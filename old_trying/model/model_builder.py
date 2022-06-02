# commenting the original btad for unet based real/fake 
from collections import OrderedDict
import torch
from .Unet import *

def freeze_bn(net):
    for module in net.modules():
        if isinstance(module, torch.nn.modules.BatchNorm2d):
            for i in module.parameters():
                i.requires_grad = False

def release_bnad(net):
    for module in net.modules():
        if isinstance(module, torch.nn.modules.BatchNorm2d):
            for i in module.parameters():
                i.requires_grad = True
        else:
            for i in module.parameters():
                i.requires_grad = False            

def init_model(cfg):

    model = UNet_mod(cfg.num_channels, cfg.num_classes, cfg.fake_ce).cuda()

    if cfg.restore_from != 'None':
        params = torch.load(cfg.restore_from)
        model.load_state_dict(params)
        print('----------Model initialize with weights from-------------: {}'.format(cfg.restore_from))

    if cfg.multigpu:
        model = nn.DataParallel(model)
    if cfg.train:
        model.train().cuda()
        print('Mode --> Train')
    else:
        model.eval().cuda()
        print('Mode --> Eval')
    
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad)) 105344 # for only batch normalisation
    
    return model

