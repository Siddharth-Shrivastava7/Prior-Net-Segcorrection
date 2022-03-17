# from . import *
import torch 
import torch.nn as nn
from .pspnet import PSPNet
from .deeplab import Deeplab
from .refinenet import RefineNet
from .relighting import LightNet, L_TV, L_exp_z, SSIM
from .discriminator import FCDiscriminator
from .loss import StaticLoss

def pred(num_classes,  model, restore_from, restore_light_path):
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda")
    # print('&&&&&&&&&&&&&&&&&&&&&')
    # print(model)
    
    if model == 'PSPNet':
        model = PSPNet(num_classes=num_classes)
    if model == 'DeepLab':
        model = Deeplab(num_classes=num_classes)
    if model == 'RefineNet':
        model = RefineNet(num_classes=num_classes, imagenet=False)

    # if restore_from == '/home/sidd_s/scratch/saved_models_hpc/saved_models/DANNet/dannet_psp.pth':
    #     print('yes')

    # print('&&&&&&&&&&&&&&&&&&&&&')
    # print(model)
    # print(restore_from)
    saved_state_dict = torch.load(restore_from)
    # saved_state_dict = torch.load('/home/sidd_s/scratch/saved_models_hpc/saved_models/DANNet/dannet_psp.pth')
    # print('&&&&&&&&&&&&&&&&&&&&&')
    model_dict = model.state_dict()
    # print('&&&&&&&&&&&&&&&&&&&&&')
    saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_dict}
    model_dict.update(saved_state_dict)
    model.load_state_dict(saved_state_dict)

    lightnet = LightNet()
    saved_state_dict = torch.load(restore_light_path)
    model_dict = lightnet.state_dict()
    saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_dict}
    model_dict.update(saved_state_dict)
    lightnet.load_state_dict(saved_state_dict)

    model = model.to(device)
    lightnet = lightnet.to(device)
    # model.eval()
    # lightnet.eval()

    weights = torch.log(torch.FloatTensor(
        [0.36869696, 0.06084986, 0.22824049, 0.00655399, 0.00877272, 0.01227341, 0.00207795, 0.0055127, 0.15928651,
         0.01157818, 0.04018982, 0.01218957, 0.00135122, 0.06994545, 0.00267456, 0.00235192, 0.00232904, 0.00098658,
         0.00413907])).cuda()
    std = 0.16 ## original
    weights = (torch.mean(weights) - weights) / torch.std(weights) * std + 1.0

    return model, lightnet, weights
    

    