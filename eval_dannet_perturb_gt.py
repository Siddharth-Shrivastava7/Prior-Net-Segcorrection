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
from dataset.network.dannet_pred import pred
from utils.optimize import *
import argparse
import network

parser = argparse.ArgumentParser() 

parser.add_argument("--gpu_id", type=str, default='5',
                        help="GPU ID") 
available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
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
parser.add_argument("--val_data_list", '-val_dr_lst', type=str, default='./dataset/list/acdc/acdc_valrgb.txt',
                        choices = ['./dataset/list/acdc/acdc_valrgb.txt'],help='list of names of validation files')  
parser.add_argument("--worker", type=int, default=4)   
parser.add_argument("--val_dataset", '-val_nm', type=str, default='synthetic_manual_acdc_val_label',
                        choices = ['acdc_val_label', 'synthetic_manual_acdc_val_label', 'dannet_pred'], help='valditation datset name') 
parser.add_argument("--print_freq", type=int, default=1)  
parser.add_argument("--epochs", type=int, default=500)   # may be req 
parser.add_argument("--snapshot", type=str, default='../scratch/saved_models/acdc/dannet',
                        help='model saving directory')   
parser.add_argument("--exp", '-e', type=int, default=1, help='which exp to run')                     
parser.add_argument("--method_eval", type=str, default='dannet_perturb_gt',
                        help='evalutating technique') 
parser.add_argument("--synthetic_perturb", type=str, default='synthetic_new_manual_dannet_20n_100p',
                        help='evalutating technique')
parser.add_argument("--color_map", action='store_true', default=False,
                        help="color_mapping project evalutation")
parser.add_argument("--direct", action='store_true', default=False,
                        help="same colormap as that of cityscapes where perturbation is taking place")
parser.add_argument("--save_path", type=str, default='synthetic_new_manual_dannet_20n_100p',
                        help='evalutating technique')
parser.add_argument("--norm",action='store_true', default=False,
                        help="normalisation of the image")
parser.add_argument("--ignore_classes", nargs="+", default=[],  
                        help="ignoring classes...blacking out those classes, generally [5,7,11,12,17]")
parser.add_argument("--posterior",action='store_true', default=False,
                        help="if posterior calculation required or not")  

args = parser.parse_args()

## dataset init 

def init_val_data(args): 
    valloader = data.DataLoader(
            BaseDataSet(args, args.val_data_dir, args.val_data_list, args.val_dataset, args.num_classes, ignore_label=255, set='val'),
            batch_size=1, shuffle=True, num_workers=args.worker, pin_memory=True)
    return valloader    

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
        if self.dataset == 'acdc_train_label' or self.dataset == 'acdc_val_label':  
            
            for name in self.img_ids: 
                img_file = os.path.join(self.root, name) 
                replace = (("_rgb_anon", "_gt_labelTrainIds"), ("acdc_trainval", "acdc_gt"), ("rgb_anon", "gt"))
                nm = name
                for r in replace: 
                    nm = nm.replace(*r) 
                label_file = os.path.join(self.root, nm)    
                self.files.append({
                        "img": img_file,
                        "label":label_file,
                        "name": name
                    }) 
        
        elif 'synthetic' in self.dataset: 
            for name in self.img_ids: 
                img_file = os.path.join(self.root, name.split('night')[0], 'night/synthetic', self.set, args.synthetic_perturb, name.split('/')[-1]) 
                replace = (("_rgb_anon", "_gt_labelTrainIds"), ("acdc_trainval", "acdc_gt"), ("rgb_anon", "gt"))
                nm = name
                for r in replace: 
                    nm = nm.replace(*r) 
                label_file = os.path.join(self.root, nm)    
                self.files.append({
                        "img": img_file,
                        "label":label_file,
                        "name": name
                    })  
                
        elif 'dannet' in self.dataset: 
            for name in self.img_ids: 
                img_file = os.path.join(self.root, name.split('night')[0], 'night/dannet_pred/val',  name.split('/')[-1])
                replace = (("_rgb_anon", "_gt_labelTrainIds"), ("acdc_trainval", "acdc_gt"), ("rgb_anon", "gt"))
                nm = name
                for r in replace: 
                    nm = nm.replace(*r) 
                label_file = os.path.join(self.root, nm)    
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
        
        ## used in transforms 
        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  ## image net mean and std  
    
        try:  
            if self.dataset in ['acdc_train_label', 'acdc_val_label']: 
                
                ## image for transforming  
                transforms_compose_img = transforms.Compose([
                    transforms.Resize((540, 960)),
                    transforms.ToTensor(),
                    transforms.Normalize(*mean_std), ## use only in training 
                ])
                image = Image.open(datafiles["img"]).convert('RGB') 
                image = transforms_compose_img(image)    
                image = torch.tensor(np.array(image)).float() 
            
                transforms_compose_label = transforms.Compose([
                        transforms.Resize((1080,1920), interpolation=Image.NEAREST)])  
                label = Image.open(datafiles["label"]) 
                label = transforms_compose_label(label)   
                label = torch.tensor(np.array(label))  
            elif 'synthetic' or 'dannet' in self.dataset:
                image = Image.open(datafiles["img"]).convert('RGB') 
                if self.args.norm: 
                    transforms_compose_img = transforms.Compose([ 
                                transforms.ToTensor(),
                                transforms.Normalize(*mean_std) ## use only in training 
                            ]) 
                    image_t = transforms_compose_img(image)   
                    image_t = torch.tensor(np.array(image_t)).float() 
                    image = torch.tensor(np.array(image)).float()
                    
                else: 
                    image = torch.tensor(np.array(image)).float()
                
                label = Image.open(datafiles["label"]) 
                label = torch.tensor(np.array(label))
                
                if self.args.norm: 
                    return image_t, image, label, name
                    
                
                return image, label, name
                                          
        except: 
            print('**************') 
            print(index)
            index = index - 1 if index > 0 else index + 1 
            return self.__getitem__(index) 

        return image, label, name
                

def init_model(args): 
    # Deeplabv3+ model for prior net
    model = network.modeling.__dict__[args.model](num_classes=args.num_classes, output_stride=args.output_stride, num_ip_channels = args.num_ip_channels )   
    network.utils.set_bn_momentum(model.backbone, momentum=0.01)  
    if args.multigpu:
        model = nn.DataParallel(model)
    params = torch.load(os.path.join('/home/sidd_s/scratch/saved_models/acdc/dannet/',args.restore_from))
    model.load_state_dict(params)
    print('----------Model initialize with weights from-------------: {}'.format(args.restore_from))

    model.eval().cuda()
    print('Mode --> Eval') 
    
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
    4: [100, 51, 43], 5: [125, 175, 175], 6: [250, 170, 30], 7: [220, 220,   0], 
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
    img_synthetic = img_synthetic.permute(1,2,0)
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
            # print('******************') 
            
            if args.val_dataset == 'acdc_val_label':
                image , seg_label, _ = batch  ## chg 
                # print(label.shape) # torch.Size([1, 1080, 1920]) 
                interp = nn.Upsample(size=(1080, 1920), mode='bilinear', align_corners=True)
                # dannet prediction 
                da_model = da_model.eval()
                lightnet = lightnet.eval()

                with torch.no_grad(): 
                    r = lightnet(image.cuda())
                    enhancement = image.cuda() + r
                    if model == 'RefineNet':
                        output2 = da_model(enhancement)
                    else:
                        _, output2 = da_model(enhancement)

                weights_prob = weights.expand(output2.size()[0], output2.size()[3], output2.size()[2], 19)
                weights_prob = weights_prob.transpose(1, 3)
                output2 = output2 * weights_prob

                output2 = interp(output2).cpu().numpy() 
                seg_tensor = torch.tensor(output2)   
                seg_pred = F.softmax(seg_tensor, dim=1)  
                
                # print(seg_tensor.shape) # torch.Size([1, 19, 1080, 1920]) 
                if args.method_eval == 'model_onehot':
                    dannet_label = seg_tensor.argmax(dim=1).float().squeeze() ## prior   
                    dannet_label[dannet_label==255] = 19  
                    label_perturb_tensor = F.one_hot(dannet_label.to(torch.int64), 20) 
                    label_perturb_tensor = label_perturb_tensor[:, :, :19]  
                    label_perturb_tensor = label_perturb_tensor.unsqueeze(dim=0).transpose(3,2).transpose(2,1)
                    
                elif args.method_eval=='shuffle' or args.method_eval == 'shuffle_w_weights': 
                    randx_pix = np.random.randint(seg_tensor.shape[2]-10)
                    randy_pix = np.random.randint(seg_tensor.shape[2]-10)
                
                    for x in range(10):
                        for y in range(10):  
                            org_batch_val = seg_tensor[:, :, randx_pix + x, randy_pix + y]  
                            shuffle_inds = torch.randperm(org_batch_val.shape[1]) 
                            seg_tensor[:, :, randx_pix + x, randy_pix + y] = org_batch_val[:, shuffle_inds] 
            
                    label_perturb_tensor = seg_tensor.detach().clone()  
                
                elif args.method_eval=='straight':
                    label_perturb_tensor = seg_tensor.detach().clone()
                
                elif args.method_eval=='straight_color':
                    label_perturb_tensor = seg_tensor.detach().clone() 
                
                elif args.method_eval=='dannet_perturb_gt':
                    ## converting the gt seglabel with train ids into onehot matrix 
                    seg_label = [seg_label[bt] for bt in range(seg_label.shape[0])]  
                    for label in seg_label:
                        label[label==255] = 19 
                    seg_label = torch.stack(seg_label, dim=0)
                    seg_pred = torch.argmax(seg_pred, dim=1)  
                    
                    for _ in range(args.num_patch_perturb): # number of patches to be cropped
                        randx = np.random.randint(seg_pred.shape[1]-args.patch_size)
                        randy = np.random.randint(seg_pred.shape[2]-args.patch_size)    
                        seg_label[:, randx: randx + args.patch_size, randy:randy+args.patch_size]  \
                                    = seg_pred[:, randx: randx + args.patch_size, randy:randy+args.patch_size]
                    
                    if args.channel19: 
                        ## gt perturb one hot 
                        gt_onehot_batch = F.one_hot(seg_label.to(torch.int64), 20) 
                        gt_onehot_batch = gt_onehot_batch.permute(0, 3, 1, 2) 
                        gt_onehot_batch = gt_onehot_batch[:, :19, :, :]  
                        label_perturb_tensor = gt_onehot_batch.detach().clone() # dont backprop on this   
                    else: 
                        gt_3ch = [torch.tensor(label_img_to_color(seg_label[bt])) for bt in range(seg_label.shape[0])]   
                        gt_3ch = torch.stack(gt_3ch, dim=0)
                        label_perturb_tensor = gt_3ch.permute(0, 3, 1, 2).detach().clone() 
                
                output =  model(label_perturb_tensor.float().cuda())       
                perturb = label_perturb_tensor.cuda() 
                seg_label = seg_label.cuda()
                output = output.squeeze()  
                perturb = perturb.squeeze()
                C,H,W = output.shape    
                
            elif 'synthetic' or 'dannet' in args.val_dataset: 
                if args.norm:  
                    image_t, image, seg_label, _ = batch  ## chg 
                    image = image.permute(0, 3, 1, 2) 
                    synthetic_input_perturb = image_t.detach().clone() 
                else: 
                    image , seg_label, _ = batch  ## chg 
                    synthetic_input_perturb = image.permute(0, 3, 1, 2).detach().clone()     
                
                if args.ignore_classes: 
                    # masking  
                    ignore_classes = json.loads(args.ignore_classes[0])  
                    ip_label = torch.tensor(color_to_label(synthetic_input_perturb[0], args)) 
                    ip_label_clone = ip_label.detach().clone() 
                    for ig_cl in ignore_classes:
                        ip_label[ip_label==ig_cl] = 255 
                    synthetic_input_perturb = label_img_to_color(ip_label)  
                    synthetic_input_perturb = torch.tensor(synthetic_input_perturb).unsqueeze(dim=0).permute(0,3,1,2)
                
                ## output from priornet 
                output =  model(synthetic_input_perturb.float().cuda())  
                if args.norm: 
                    perturb = image.detach().clone().cuda()
                else:     
                    if args.ignore_classes:
                    ## unmasking for reverting input
                        ip_label = torch.tensor(color_to_label(synthetic_input_perturb[0], args)) 
                        for ig_cl in ignore_classes: 
                            ip_label[ip_label_clone==ig_cl] = ig_cl 
                        synthetic_input_perturb = label_img_to_color(ip_label)   
                        synthetic_input_perturb = torch.tensor(synthetic_input_perturb).unsqueeze(dim=0).permute(0,3,1,2)     
                        
                    perturb = synthetic_input_perturb.cuda() 
                seg_label = seg_label.cuda()
                output = output.squeeze()  
                perturb = perturb.squeeze()
                C,H,W = output.shape 
           
            #########################################################################corrected one 
            # if args.ignore_classes: 
            #     Mask = (seg_label.squeeze())<C  
            #     ignore_classes = json.loads(args.ignore_classes[0]) 
            #     for ig_cl in ignore_classes: 
            #         Mask = Mask * ((seg_label.squeeze())!= ig_cl) 
            # else:
            #     pass  
            
            Mask = (seg_label.squeeze())<C  # it is ignoring all the labels values equal or greater than 19 #(1080, 1920)  
                
            pred_e = torch.linspace(0,C-1, steps=C).view(C, 1, 1)  
            pred_e = pred_e.repeat(1, H, W).cuda()  
            pred = output.argmax(dim=0).float() ## prior   
            
            if args.ignore_classes:
            ## unmasking 
                for ig_cl in ignore_classes:
                    pred[ip_label_clone==ig_cl] = ig_cl 
            
            pred_mask = torch.eq(pred_e, pred).byte()    
            pred_mask = pred_mask*Mask 
        
            label_e = torch.linspace(0,C-1, steps=C).view(C, 1, 1)
            label_e = label_e.repeat(1, H, W).cuda()
            seg_label = seg_label.view(1, H, W)
            label_mask = torch.eq(label_e, seg_label.float()).byte()
            label_mask = label_mask*Mask
            # print(label_mask.shape) # torch.Size([19, 1080, 1920])

            tmp_inter = label_mask+pred_mask
            # print(tmp_inter.shape) # torch.Size([19, 1080, 1920])
            # print(torch.unique(tmp_inter)) # tensor([0, 1, 2])
            res = ((tmp_inter>0) * (pred_mask==1)).int() 
            # print(torch.unique(res)) 
            # print(torch.where(res)) 
            # print((res == 1).nonzero(as_tuple=True)) 
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

            # pred = perturb.argmax(dim=0).float() ## perturb  
            ## synthetic image offline uploading   
            pred = torch.tensor(color_to_label(perturb, args)).cuda()
            pred_mask2 = torch.eq(pred_e, pred).byte()    
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
        
        # breakpoint()
        print('loss calc and saving images')        
        ## not calculating loss for now  
        for i_iter, batch in tqdm(enumerate(testloader)): 
                
            if self.args.val_dataset == 'acdc_val_label':
                image, seg_label, name = batch
                nm = name[0].split('/')[-1]
                ## perturbation 
                with torch.no_grad():
                    r = self.lightnet(image.cuda())  
                    enhancement = image.cuda() + r
                    if self.args.model_dannet == 'RefineNet':
                        output2 = self.da_model(enhancement)
                    else:
                        _, output2 = self.da_model(enhancement)

                ## weighted cross entropy for which they have been used so here not using it in the training so ignoring for now   
                weights_prob = self.weights.expand(output2.size()[0], output2.size()[3], output2.size()[2], 19)
                weights_prob = weights_prob.transpose(1, 3) 
                output2 = output2 * weights_prob 

                output2 = self.interp(output2).cpu().numpy() 
                seg_pred = torch.tensor(output2) 
                seg_pred = F.softmax(seg_pred, dim=1) 
                
                ## converting the gt seglabel with train ids into onehot matrix 
                seg_label = [seg_label[bt] for bt in range(seg_label.shape[0])]  
                for label in seg_label:
                    label[label==255] = 19 
                seg_label = torch.stack(seg_label, dim=0)
                seg_pred = torch.argmax(seg_pred, dim=1)  
                
                for _ in range(self.args.num_patch_perturb): # number of patches to be cropped
                    randx = np.random.randint(seg_pred.shape[1]-self.args.patch_size)
                    randy = np.random.randint(seg_pred.shape[2]-self.args.patch_size)    
                    seg_label[:, randx: randx + self.args.patch_size, randy:randy+self.args.patch_size]  \
                                = seg_pred[:, randx: randx + self.args.patch_size, randy:randy+self.args.patch_size]
                
                if self.args.channel19:
                    ## gt perturb one hot 
                    gt_onehot_batch = F.one_hot(seg_label.to(torch.int64), 20) 
                    gt_onehot_batch = gt_onehot_batch.permute(0, 3, 1, 2) 
                    gt_onehot_batch = gt_onehot_batch[:, :19, :, :]  
                    seg_pred_tensor = gt_onehot_batch.detach().clone() # dont backprop on this   
                else: 
                    gt_3ch = [torch.tensor(label_img_to_color(seg_label[bt])) for bt in range(seg_label.shape[0])]   
                    gt_3ch = torch.stack(gt_3ch, dim=0)
                    seg_pred_tensor = gt_3ch.permute(0, 3, 1, 2).detach().clone()  
            
            elif 'synthetic' or 'dannet' in self.args.val_dataset:     
                if self.args.norm:  
                    image_t, image, seg_label, name = batch  ## chg 
                    image = image.permute(0, 3, 1, 2) 
                    synthetic_input_perturb = image_t.detach().clone() 
                else: 
                    image , seg_label, name = batch  ## chg 
                    synthetic_input_perturb = image.permute(0, 3, 1, 2).detach().clone()                         
                    # image = image.permute(0, 3, 1, 2).detach().clone() 
                nm = name[0].split('/')[-1]  
                
                if args.ignore_classes:
                    # masking 
                    ignore_classes = json.loads(args.ignore_classes[0])  
                    ip_label = torch.tensor(color_to_label(synthetic_input_perturb[0], args))   
                    ip_label_clone = ip_label.detach().clone()
                    for ig_cl in ignore_classes:
                        ip_label[ip_label==ig_cl] = 255 
                    synthetic_input_perturb = label_img_to_color(ip_label)  
                    synthetic_input_perturb = torch.tensor(synthetic_input_perturb).unsqueeze(dim=0).permute(0,3,1,2)     
            
            ## changing name    
            seg_correct = self.model(synthetic_input_perturb.float().cuda())    
            
            # seg_preds = torch.argmax(label_perturb_tensor, dim=1) 
            # pertub dannet gt
            # seg_preds = torch.tensor(color_to_label(image[0], self.args))  
            ## ignore classes blacking out  
            # if self.args.ignore_classes:   
            #     ignore_classes = json.loads(args.ignore_classes[0]) 
            #     ip_imglb = color_to_label(image[0], self.args)
            #     for ig_cl in ignore_classes: 
            #         seg_preds[ip_imglb==ig_cl] = 19 # black region  
            
            seg_preds = torch.tensor(color_to_label(synthetic_input_perturb[0], self.args))   
            if args.ignore_classes:
                ## unmasking for reverting input
                for ig_cl in ignore_classes: 
                    seg_preds[ip_label_clone==ig_cl] = ig_cl
            
            seg_preds[seg_label[0]==255] = 19 # ignore black region             
            seg_preds =  label_img_to_color(seg_preds)  

            ## correction output              
            seg_corrects = torch.argmax(seg_correct, dim=1)[0] 
            
            if self.args.ignore_classes:   
                ## unmasking 
                for ig_cl in ignore_classes:  
                    seg_corrects[ip_label_clone==ig_cl] = ig_cl   
            
            seg_corrects[seg_label[0]==255] = 19 # ignore black region
            seg_corrects = label_img_to_color(seg_corrects) 
            
            ## seg label 
            seg_label[0][seg_label[0]==255] = 19
            seg_label_arr = label_img_to_color(seg_label[0]) 
            
            if args.color_map:  
                ## manual color map
                pred = torch.tensor(color_to_label(image[0], args)) 
                pred[seg_label[0]==255] = 19 # ignore black region  
                pred =  label_img_to_manual_color(pred)
            else:
                ## dannet output 
                path = os.path.join('/home/sidd_s/scratch/dataset/acdc_trainval/rgb_anon/night/dannet_pred/val', nm) 
                pred = np.array(Image.open(path))             
            
            ## collague way of saving images 
            imgh1 = np.hstack((seg_label_arr, seg_preds))
            imgh2 = np.hstack((pred, seg_corrects))
            imgg = Image.fromarray(np.vstack((imgh1, imgh2))) 
            if not os.path.exists(os.path.join('collague', self.args.save_path)):
                os.makedirs(os.path.join('collague', self.args.save_path)) 
            path_col = os.path.join('collague', self.args.save_path, nm)
            imgg.save(path_col)
            
            seg_label = seg_label.long().cuda() # cross entropy    
            loss = CrossEntropy2d() # ce loss  
            seg_loss = loss(seg_correct, seg_label) 
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
    if args.multigpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = "2,3,4" 
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id 
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

