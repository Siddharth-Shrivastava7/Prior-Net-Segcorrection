import datetime
import os
import random
import sys
import time
from turtle import pd

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from easydict import EasyDict as edict
from PIL import Image
from torch.utils import data
from tqdm import tqdm
import json 
# from dataset.network.dannet_pred import pred 
from utils.optimize import *
import argparse
import network 
from dataset.network.pspnet import PSPNet
from dataset.network.deeplab import Deeplab
from dataset.network.refinenet import RefineNet
from dataset.network.relighting import LightNet, L_TV, L_exp_z, SSIM
from dataset.network.discriminator import FCDiscriminator
from dataset.network.loss import StaticLoss


parser = argparse.ArgumentParser() 
parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID") 
available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
available_models.append('unet')
parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16]) 
parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name') 
parser.add_argument("--num_classes", type=int, default=19,
                        help="num classes (default: 19)")  
parser.add_argument("--num_ip_channels", '-ip_chs',type=int, default=3,
                        help="num of input channels (default: 3)")  
parser.add_argument("--channel19", '-ch19',action='store_true', default=False,
                        help="to use colored o/p")   # may be req 
parser.add_argument("--restore_from", '-res_f', type=str, default='deeplabv3+_mobilenet_adam_lr3e-4_endlr1e-5____w_pretrain_bt8.pth',
                        help='prior net trained model path') 
parser.add_argument("--multigpu", '-mul',action='store_true', default=False,
                        help="multigpu training") 
parser.add_argument("--model_dannet", type=str, default='PSPNet',
                        help='seg model of dannet') 
parser.add_argument("--restore_from_da", type=str, default='/home/sidd_s/scratch/saved_models/DANNet/dannet_psp.pth', help='Dannet pre-trained model path') 
parser.add_argument("--restore_light_path", type=str, default='/home/sidd_s/scratch/saved_models/DANNet/dannet_psp_light.pth',
                        help='Dannet pre-trained light net model path') 
parser.add_argument("--num_patch_perturb", type=int, default=100,  # may be req 
                        help="num of patches to be perturbed") 
parser.add_argument("--patch_size", type=int, default=10, # may be req 
                        help="size of square patch to be perturbed")
parser.add_argument("--val_data_dir", '-val_dr', type=str, default='/home/sidd_s/scratch/dataset',
                        help='train data directory') 
parser.add_argument("--val_data_list", '-val_dr_lst', type=str, default='/home/sidd_s/Prior-Net-Segcorrection/dataset/list/acdc/acdc_valrgb.txt',
                        choices = ['/home/sidd_s/Prior-Net-Segcorrection/dataset/list/acdc/acdc_valrgb.txt'],help='list of names of validation files')  
parser.add_argument("--worker", type=int, default=4)   
parser.add_argument("--val_dataset", '-val_nm', type=str, default='synthetic_manual_acdc_val_label',
                        choices = ['acdc_val_label', 'synthetic_manual_acdc_val_label', 'dannet_pred'], help='valditation datset name') 
parser.add_argument("--print_freq", type=int, default=1)  
parser.add_argument("--epochs", type=int, default=500)   # may be req 
parser.add_argument("--snapshot", type=str, default='/home/sidd_s/scratch/saved_models/acdc/dannet',
                        help='model saving directory')   
parser.add_argument("--exp", '-e', type=int, default=1, help='which exp to run')                     
parser.add_argument("--method_eval", type=str, default='dannet_perturb_gt',
                        help='evalutating technique') 
parser.add_argument("--synthetic_perturb", type=str, default='synthetic_manual_dannet_50n_100p_1024im',
                        help='evalutating technique')
parser.add_argument("--color_map", action='store_true', default=False,
                        help="color_mapping project evalutation")
parser.add_argument("--direct", action='store_true', default=False,
                        help="same colormap as that of cityscapes where perturbation is taking place")
parser.add_argument("--save_path", type=str, default='synthetic_new_manual_dannet_50n_100p',
                        help='evalutating technique')
parser.add_argument("--norm",action='store_true', default=False,
                        help="normalisation of the image")
parser.add_argument("--ignore_classes", nargs="+", default=[],  
                        help="ignoring classes...blacking out those classes, generally [6,7,11,12,17,18]")
parser.add_argument("--posterior",action='store_true', default=False,
                        help="if posterior calculation required or not")  
parser.add_argument("--small_model",action='store_true', default=False,
                        help="using augmentation during training")  ## denoising auto encoder (thinking of...let's seeee)
parser.add_argument("--val_perturb_path", type=str, default='1000n_20p_dannet_pred',
                        help='evalutating technique') 

args = parser.parse_args() 

if args.multigpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2" 
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id  
## dataset init 

weights = torch.log(torch.FloatTensor(
                    [0.36869696, 0.06084986, 0.22824049, 0.00655399, 0.00877272, 0.01227341, 0.00207795, 0.0055127, 0.15928651, 0.01157818, 0.04018982, 0.01218957, 0.00135122, 0.06994545, 0.00267456, 0.00235192, 0.00232904, 0.00098658, 0.00413907])).cuda()
weights = (torch.mean(weights) - weights) / torch.std(weights) * 0.16 + 1.0   ## std 0.16 for validation


def init_val_data(args): 
    valloader = data.DataLoader(
            BaseDataSet(args, args.val_data_dir, args.val_data_list, args.val_dataset, args.num_classes, ignore_label=255, set='val'),
            batch_size=1, shuffle=True, num_workers=args.worker, pin_memory=True)
    return valloader    

def pred(num_classes,  model, restore_from, restore_light_path):
    
    if model == 'PSPNet':
        model = PSPNet(num_classes=num_classes)
    if model == 'DeepLab':
        model = Deeplab(num_classes=num_classes)
    if model == 'RefineNet':
        model = RefineNet(num_classes=num_classes, imagenet=False)

    # saved_state_dict = torch.load(restore_from, map_location='cuda: 7')
    saved_state_dict = torch.load(restore_from)
    
    # saved_state_dict = torch.load('/home/sidd_s/scratch/saved_models_hpc/saved_models/DANNet/dannet_psp.pth')
    model_dict = model.state_dict()
    saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_dict}
    model_dict.update(saved_state_dict)
    model.load_state_dict(saved_state_dict)

    lightnet = LightNet()
    saved_state_dict = torch.load(restore_light_path)
    model_dict = lightnet.state_dict()
    saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_dict}
    model_dict.update(saved_state_dict)
    lightnet.load_state_dict(saved_state_dict)

    model = model.cuda()
    lightnet = lightnet.cuda() 
    
    weights = torch.log(torch.FloatTensor(
        [0.36869696, 0.06084986, 0.22824049, 0.00655399, 0.00877272, 0.01227341, 0.00207795, 0.0055127, 0.15928651,
         0.01157818, 0.04018982, 0.01218957, 0.00135122, 0.06994545, 0.00267456, 0.00235192, 0.00232904, 0.00098658,
         0.00413907])).cuda()
    std = 0.16 ## original
    weights = (torch.mean(weights) - weights) / torch.std(weights) * std + 1.0

    return model, lightnet, weights

class BaseDataSet(data.Dataset): 
    def __init__(self, args, root, list_path,dataset, num_classes, ignore_label=255, set='val'): 

        self.root = root 
        self.list_path = list_path
        self.set = set
        self.dataset = dataset 
        self.num_classes = num_classes 
        self.ignore_label = ignore_label
        self.args = args  
        self.img_ids = []  

        with open(self.list_path) as f: 
            for item in f.readlines(): 
                fields = item.strip().split('\t')[0]
                if ' ' in fields:
                    fields = fields.split(' ')[0]
                self.img_ids.append(fields)  
        
        self.files = []  
        
        for name in self.img_ids: 
            replace = (("_rgb_anon", "_gt_labelTrainIds"), ("acdc_trainval", "acdc_gt"), ("rgb_anon", "gt"))
            nm = name
            for r in replace: 
                nm = nm.replace(*r) 
            label_file = os.path.join(self.root, nm)     
            img_file = os.path.join(self.root, name.split('night')[0], 'night/dannet_pred', self.set, name.split('/')[-1])
            self.files.append({
                    "img": img_file,
                    "label":label_file,
                    "name": name
                })   
                                
    def __len__(self):
        return len(self.files) 
    
    def __getitem__(self, index): 
        datafiles = self.files[index]
        name = datafiles["name"]    
        
        transforms_compose_label_val = transforms.Compose([
                        transforms.Resize((1080,1920), interpolation=Image.NEAREST)])  
        
        try:  
            label = Image.open(datafiles["label"]) 
            label = transforms_compose_label_val(label)  
            label = torch.tensor(np.array(label))  
            label_perturb_path = os.path.join(self.root, name.split('/val/')[0] + '/synthetic/val/' + args.val_perturb_path + '/' + name.split('/')[-1].replace('_rgb_anon.png','_gt_labelColor.png')) 
            image = np.array(Image.open(label_perturb_path))   
            
            if self.args.norm: 
                transforms_compose_img = transforms.Compose([ 
                            transforms.ToTensor()
                        ]) 
                image_t = transforms_compose_img(image)   
            else: 
                image_t = image 
            
            image_t = torch.tensor(np.array(image_t)).float() 
            image = torch.tensor(np.array(image)).float() 
            label = Image.open(datafiles["label"]) 
            label = torch.tensor(np.array(label))       
                                          
        except Exception as e: 
            print(e)

        return image_t, image, label, name
                

def init_model(args): 
    if args.small_model: 
        model = network.unet.Unet(input_channels=args.num_ip_channels, num_classes=args.num_classes, num_filters=[32,64,128,192], initializers={'w':'he_normal', 'b':'normal'})  
    else:     
        # Deeplabv3+ model for prior net
        model = network.modeling.__dict__[args.model](num_classes=args.num_classes, output_stride=args.output_stride, num_ip_channels = args.num_ip_channels )   
        network.utils.set_bn_momentum(model.backbone, momentum=0.01)  
    if args.multigpu:
        model = nn.DataParallel(model)
    params = torch.load(os.path.join('/home/sidd_s/scratch/saved_models/acdc/dannet/',args.restore_from))
    model.eval().cuda() 
    print('Mode --> Eval') 
    model.load_state_dict(params)
    print('----------Model initialize with weights from-------------: {}'.format(args.restore_from)) 
    return model

def print_iou(iou, acc, miou, macc):
    for ind_class in range(iou.shape[0]):
        print('===> {0:2d} : {1:.2%} {2:.2%}'.format(ind_class, iou[ind_class, 0].item(), acc[ind_class, 0].item()))
    print('mIoU: {:.2%} mAcc : {:.2%} '.format(miou, macc)) 
    
def label_img_to_color(img): 
    label_to_color = {
        0: [128, 64,128],
        1: [244, 35,232],
        2: [ 70, 70, 70],
        3: [102,102,156],
        4: [190,153,153],
        5: [153,153,153],
        6: [250,170, 30],
        7: [220,220,  0],
        8: [107,142, 35],
        9: [152,251,152],
        10: [ 70,130,180],
        11: [220, 20, 60],
        12: [255,  0,  0],
        13: [  0,  0,142],
        14: [  0,  0, 70],
        15: [  0, 60,100],
        16: [  0, 80,100],
        17: [  0,  0,230],
        18: [119, 11, 32],
        19: [0,  0, 0], 
        20: [255, 255, 255],
        255: [0,  0, 0] 
        } 
    img = np.array(img.cpu())
    img_height, img_width = img.shape
    img_color = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    for row in range(img_height):
        for col in range(img_width):
            label = img[row][col] 
            img_color[row, col] = np.array(label_to_color[label])   
    return img_color

def label_img_to_manual_color(img): 
    manual_dd = {
    0: [77, 77, 77], 1: [100, 110, 120], 2: [150,100,175], 3: [102, 102, 156], 
    4: [100, 51, 43], 5: [125, 175, 175], 6: [250, 170, 30], 7: [220, 220,  0], 
    8: [107, 142,  35], 9: [70, 74,  20], 10: [50, 100, 200], 11: [190, 153, 153], 12: [140,100,100], 13: [255,255,255], 14: [15, 25,  80], 15: [50, 50,  190], 16: [10, 80, 90], 17: [180, 60,  50], 18: [120, 50,  60], 19: [0,0,0]
    } # manual 
    img = np.array(img.cpu())
    img_height, img_width = img.shape
    img_color = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    for row in range(img_height):
        for col in range(img_width):
            label = img[row][col] 
            img_color[row, col] = np.array(manual_dd[label])   
    return img_color


def color_to_label(img_synthetic, args):
    if args.direct: 
        manual_dd = {
        0: [128, 64,128],
        1: [244, 35,232],
        2: [ 70, 70, 70],
        3: [102,102,156],
        4: [190,153,153],
        5: [153,153,153],
        6: [250,170, 30],
        7: [220,220,  0],
        8: [107,142, 35],
        9: [152,251,152],
        10: [ 70,130,180],
        11: [220, 20, 60],
        12: [255,  0,  0],
        13: [  0,  0,142],
        14: [  0,  0, 70],
        15: [  0, 60,100],
        16: [  0, 80,100],
        17: [  0,  0,230],
        18: [119, 11, 32],
        19: [0,  0, 0], 
        20: [255, 255, 255],
        255: [0,  0, 0] 
        } # original labels when used
    else: 
        manual_dd = {
        0: [77, 77, 77], 1: [100, 110, 120], 2: [150,100,175], 3: [102, 102, 156], 
        4: [100, 51, 43], 5: [125, 175, 175], 6: [250, 170, 30], 7: [220, 220,   0], 
        8: [107, 142,  35], 9: [70, 74,  20], 10: [50, 100, 200], 11: [190, 153, 153], 12: [140,100,100], 13: [255,255,255], 14: [15, 25,  80], 15: [50, 50,  190], 16: [10, 80, 90], 17: [180, 60,  50], 18: [120, 50,  60], 19: [0,0,0]
        } # manual 
    
    inv_manual_dd = {str(v): k for k, v in manual_dd.items()}
    img = np.array(img_synthetic.cpu())
    img_height, img_width, _ = img.shape
    img_label = np.zeros((img_height, img_width), dtype=np.uint8)  
    for row in range(img_height):
        for col in range(img_width):
            img_label[row, col] = np.array(inv_manual_dd[str(img[row,col].astype('int64').tolist())])  

    return img_label
     

class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255, real_label=100):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label
        self.real_label = real_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label) * (target!= self.real_label) 
        target = target[target_mask]
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        # loss = F.nll_loss(torch.log(predict), target, weight=weight, size_average=self.size_average) ## NLL loss cause the pred is now in softmax form..
        return loss 
    
def compute_iou(model, testloader, args, da_model, lightnet, weights):
    model = model.eval()

    interp = nn.Upsample(size=(1080,1920), mode='bilinear', align_corners=True)   # dark_zurich -> (1080,1920) 
    union = torch.zeros(args.num_classes, 1,dtype=torch.float).cuda().float()
    inter = torch.zeros(args.num_classes, 1, dtype=torch.float).cuda().float()
    preds = torch.zeros(args.num_classes, 1, dtype=torch.float).cuda().float() 
    # extra 
    # gts = torch.zeros(args.num_classes, 1, dtype=torch.float).cuda().float() 

    union2 = torch.zeros(args.num_classes, 1,dtype=torch.float).cuda().float()
    inter2 = torch.zeros(args.num_classes, 1, dtype=torch.float).cuda().float()
    preds2 = torch.zeros(args.num_classes, 1, dtype=torch.float).cuda().float()
    # extra 
    # gts2 = torch.zeros(args.num_classes, 1, dtype=torch.float).cuda().float() 

    with torch.no_grad():
        for _, batch in tqdm(enumerate(testloader)):  
            image_t ,image , seg_label, _ = batch      
            
            breakpoint() 
    
            if args.ignore_classes: 
                # masking  
                ignore_classes = json.loads(args.ignore_classes[0])  
                ip_label = torch.tensor(color_to_label(image[0], args))   
            
            output =  model(image_t.float().cuda())  

            perturb = image.detach().clone().cuda() 
            seg_label = seg_label.cuda()
            output = output.squeeze()  
            perturb = perturb.squeeze()
            C,H,W = output.shape 
            
            Mask = (seg_label.squeeze())<C  #it is ignoring all the labels values equal or greater than 19 #(1080, 1920)  
            pred_e = torch.linspace(0,C-1, steps=C).view(C, 1, 1)  
            pred_e = pred_e.repeat(1, H, W).cuda()  
            pred = output.argmax(dim=0).float() 
    
            if args.ignore_classes:
            ## restoring the value of ignore classes +
                for ig_cl in ignore_classes:
                    pred[ip_label==ig_cl] = ig_cl 

            pred_mask = torch.eq(pred_e, pred).byte()    
            pred_mask = pred_mask*Mask 
            label_e = torch.linspace(0,C-1, steps=C).view(C, 1, 1)
            label_e = label_e.repeat(1, H, W).cuda()
            seg_label = seg_label.view(1, H, W)
            label_mask = torch.eq(label_e, seg_label.float()).byte()
            label_mask = label_mask*Mask
            # print(label_mask.shape) # torch.Size([19, 1080, 1920])

            tmp_inter = label_mask+pred_mask 
            cu_inter = (tmp_inter==2).view(C, -1).sum(dim=1, keepdim=True).float()
            cu_union = (tmp_inter>0).view(C, -1).sum(dim=1, keepdim=True).float()
            cu_preds = pred_mask.view(C, -1).sum(dim=1, keepdim=True).float()
            # extra
            # cu_gts = label_mask.view(C, -1).sum(dim=1, keepdim=True).float()
            # gts += cu_gts
            union+=cu_union
            inter+=cu_inter
            preds+=cu_preds
            ##########################################################################corrected one 

            #########################################################################perturbed one 

            ## synthetic image offline uploading   
            pred2 = torch.tensor(color_to_label(perturb, args)).cuda()
            pred_mask2 = torch.eq(pred_e, pred2).byte()    
            pred_mask2 = pred_mask2*Mask 

            tmp_inter2 = label_mask+pred_mask2
            # print(tmp_inter.shape)  # torch.Size([19, 1080, 1920])
            cu_inter2 = (tmp_inter2==2).view(C, -1).sum(dim=1, keepdim=True).float()
            cu_union2 = (tmp_inter2>0).view(C, -1).sum(dim=1, keepdim=True).float()
            cu_preds2 = pred_mask2.view(C, -1).sum(dim=1, keepdim=True).float()
            # extra
            # cu_gts2 = label_mask.view(C, -1).sum(dim=1, keepdim=True).float()
            # gts += cu_gts
            union2+=cu_union2
            inter2+=cu_inter2
            preds2+=cu_preds2
            #########################################################################perturbed one

        ###########original
        print("perturbed one")
        iou2 = inter2/union2
        acc2 = inter2/preds2
        iou2 = np.array(iou2.cpu())
        iou2 = torch.tensor(np.where(np.isnan(iou2), 0, iou2))
        acc2 = np.array(acc2.cpu())
        acc2 = torch.tensor(np.where(np.isnan(acc2), 0, acc2))

        mIoU2 = iou2.mean().item()
        mAcc2 = acc2.mean().item() 
        print('====Perturbed====')
        print_iou(iou2, acc2, mIoU2, mAcc2)
        ##################


        ###########original
        print("corrected one")
        iou = inter/union
        acc = inter/preds 
        iou = np.array(iou.cpu())
        iou = torch.tensor(np.where(np.isnan(iou), 0, iou))
        acc = np.array(acc.cpu())
        acc = torch.tensor(np.where(np.isnan(acc), 0, acc))

        mIoU = iou.mean().item()   
        mAcc = acc.mean().item() 
        print('====Prior Net====')
        print_iou(iou, acc, mIoU, mAcc)  ## original
        ################## 

        return iou, mIoU, acc, mAcc


class BaseTrainer(object):
    def __init__(self):
       super().__init__() 
       pass 
       
    def validate(self): 
        self.model = self.model.eval() 
        total_loss = 0
        testloader = init_val_data(self.args) 
        print('In validation') 
        
        print("MIou calculation: ")    
        iou, mIoU, acc, mAcc = compute_iou(self.model, testloader, self.args, self.da_model, self.lightnet, self.weights)
        
        print('loss calc and saving images')        
        ## not calculating loss for now  
        for i_iter, batch in tqdm(enumerate(testloader)): 
            image_t ,image , seg_label, name = batch   
            nm = name[0].split('/')[-1] 

            if args.ignore_classes: 
                # preserving ignore classes initial values
                ignore_classes = json.loads(args.ignore_classes[0])  
                ip_label = torch.tensor(color_to_label(image[0], args))   
            
            output =  self.model(image_t.float().cuda())   
            seg_label = seg_label.cuda()
            
            ## seg correct
            seg_correct = output[0].argmax(dim=0).float()  
            if args.ignore_classes:
            ## restoring the value of ignore classes +
                for ig_cl in ignore_classes:
                    seg_correct[ip_label==ig_cl] = ig_cl  
            seg_correct[seg_label[0]==255] = 19 # ignore black region 
            seg_correct = seg_correct.detach()            
            seg_correct =  label_img_to_color(seg_correct)   ## correction
            
            ## seg label 
            seg_label[0][seg_label[0]==255] = 19
            seg_label_arr = label_img_to_color(seg_label[0]) 

            ## dannet output 
            path = os.path.join('/home/sidd_s/scratch/dataset/acdc_trainval/rgb_anon/night/dannet_pred/val', nm) 
            pred = np.array(Image.open(path))        
            
            ## seg perturb 
            perturb = np.array(image[0]) 
            perturb = perturb.astype(np.uint8) 

            ## collague way of saving images 
            imgh1 = np.hstack((seg_label_arr, perturb))
            imgh2 = np.hstack((pred, seg_correct))
            imgg = Image.fromarray(np.vstack((imgh1, imgh2))) 
            if not os.path.exists(os.path.join('/home/sidd_s/Prior-Net-Segcorrection/collague', self.args.save_path)):
                os.makedirs(os.path.join('/home/sidd_s/Prior-Net-Segcorrection/collague', self.args.save_path)) 
            path_col = os.path.join('/home/sidd_s/Prior-Net-Segcorrection/collague', self.args.save_path, nm) 
            # print(path_col)
            imgg.save(path_col)
            
            seg_label = seg_label.long().cuda() # cross entropy    
            loss = CrossEntropy2d() # ce loss  
            seg_loss = loss(output, seg_label) 
            total_loss += seg_loss.item()   

        total_loss /= len(iter(testloader))
        print('---------------------')
        print('Validation seg loss: {}'.format(total_loss))   
          
        return total_loss

    
class Trainer(BaseTrainer):
    def __init__(self, model, args): 
        super().__init__()
        self.model = model
        self.args = args
        self.da_model, self.lightnet, self.weights = pred(self.args.num_classes, self.args.model_dannet, self.args.restore_from_da, self.args.restore_light_path)  
        self.da_model = self.da_model.eval() 
        self.lightnet = self.lightnet.eval() 
        self.interp = nn.Upsample(size=(1080, 1920), mode='bilinear', align_corners=True)

    def eval(self):
        print('Lets eval...') 
        valid_loss = self.validate()    
        return     
 
def main(args):  
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    cudnn.enabled = True
    cudnn.benchmark = True   
    model = init_model(args)
    trainer = Trainer(model, args)
    trainer.eval() 


if __name__ == "__main__":  
    mp.set_start_method('spawn')   ## for different random value using np.random  
    main(args)

