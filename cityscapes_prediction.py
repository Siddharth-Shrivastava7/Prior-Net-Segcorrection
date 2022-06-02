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
from utils.optimize import *
import argparse
import network

parser = argparse.ArgumentParser() 

parser.add_argument("--gpu_id", type=str, default='0',
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
parser.add_argument("--restore_from", '-res_f', type=str, default='/home/sidd_s/scratch/saved_models/citydeeplabv3+/best_deeplabv3plus_mobilenet_cityscapes_os16.pth',
                        help='cityscapes trained model path') 
parser.add_argument("--multigpu", '-mul',action='store_true', default=False,
                        help="multigpu training") 
parser.add_argument("--val_data_dir", '-val_dr', type=str, default='/home/sidd_s/scratch/dataset/cityscapes/',
                        help='train data directory') 
parser.add_argument("--val_data_list", '-val_dr_lst', type=str, default='./dataset/list/cityscapes/val.txt',
                        help='list of names of validation files')  
parser.add_argument("--input_size", '-ip_sz', type=int, nargs="+", default=[1920, 1080],
                        help='actual input size')
available_val_datasets = ['synthetic_cityscapes_manual_color', 'synthetic_cityscapes', 'synthetic_cityscapes', 'synthetic_cityscapes_centroid_median']  
parser.add_argument("--val_dataset", '-val_nm', type=str, default='synthetic_cityscapes',
                        choices=available_val_datasets, help='valditation datset name') 
parser.add_argument("--exp", '-e', type=int, default=1, help='which exp to run')                     
parser.add_argument("--save_path", type=str, default='synthetic_mean',
                        help='saving path for file')
parser.add_argument("--worker", type=int, default=4)   

args = parser.parse_args()

## dataset init 
def init_val_data(args): 
    
    valloader = data.DataLoader(
            BaseDataSet(args, args.val_data_dir, args.val_data_list, args.val_dataset, args.num_classes, ignore_label=255, set='val'),
            batch_size=1, shuffle=True, num_workers=args.worker, pin_memory=True)

    return valloader    

class BaseDataSet(data.Dataset): 
    def __init__(self, args, root, list_path, dataset, num_classes, ignore_label=255, set='val'): 

        self.root = root 
        self.list_path = list_path
        self.dataset = dataset 
        self.num_classes = num_classes 
        self.ignore_label = ignore_label
        self.args = args  
        self.img_ids = []  
        self.set = set
        
        with open(self.list_path) as f: 
            for item in f.readlines(): 
                fields = item.strip().split('\t')[0]
                if ' ' in fields:
                    fields = fields.split(' ')[0]
                self.img_ids.append(fields)  
        
        self.files = []  
        
        # if args.val_dataset.find('synthetic')!=-1:
        if 'synthetic' in args.val_dataset:
            for name in self.img_ids:
                nm = name.split('/')[-1] 
                img_file = os.path.join(self.root, args.val_dataset, nm)    
                nm_lb = name.replace('_leftImg8bit', '_gtFine_labelIds') 
                label_file = os.path.join(self.root, 'gtFine', self.set, nm_lb)      
                self.files.append({
                        "img": img_file,
                        "label":label_file,
                        "name": name
                    }) 
        elif 'leftImg8bit' in args.val_dataset:
            for name in self.img_ids:
                img_file = os.path.join(self.root, args.val_dataset, self.set, name)    
                nm_lb = name.replace('_leftImg8bit', '_gtFine_labelIds') 
                label_file = os.path.join(self.root,'gtFine', self.set, nm_lb)    
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
    
        try:  
            mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  ## image net mean and std  
            ## image for transforming  
            transforms_compose_img = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(*mean_std), ## use only in training 
            ])
            image = Image.open(datafiles["img"]).convert('RGB') 
            image = transforms_compose_img(image)    
            image = torch.tensor(np.array(image)).float()            
            
            label = Image.open(datafiles["label"])  
            label = torch.tensor(encode_labels(np.array(label)))   # train label ids should be used for validation or training  
             
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
    params = torch.load(args.restore_from)['model_state']
    model.load_state_dict(params)
    print('----------Model initialize with weights from-------------: {}'.format(args.restore_from))
    if args.multigpu:
        model = nn.DataParallel(model)  
    model.eval().cuda()
    print('Mode --> Eval') 
    return model

def print_iou(iou, acc, miou, macc):
    for ind_class in range(iou.shape[0]):
        print('===> {0:2d} : {1:.2%} {2:.2%}'.format(ind_class, iou[ind_class, 0].item(), acc[ind_class, 0].item()))
    print('mIoU: {:.2%} mAcc : {:.2%} '.format(miou, macc)) 
    
    
def label_img_to_color(img, save_path=None): 
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
        255: [0,  0, 0],
        } 
    img = np.array(img.cpu())
    img_height, img_width = img.shape
    img_color = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    for row in range(img_height):
        for col in range(img_width):
            label = img[row][col] 
            img_color[row, col] = np.array(label_to_color[label])  
    if save_path:  
        im = Image.fromarray(img_color) 
        im.save(save_path)  
    return img_color

mapping_20 = { 
    0: 255,
    1: 255,
    2: 255,
    3: 255,
    4: 255,
    5: 255,
    6: 255,
    7: 0,
    8: 1,
    9: 255,
    10: 255,
    11: 2,
    12: 3,
    13: 4,
    14: 255,
    15: 255,
    16: 255,
    17: 5,
    18: 255,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    29: 255,
    30: 255,
    31: 16,
    32: 17,
    33: 18,
    -1: 255
}
def encode_labels(mask):
    label_mask = np.zeros_like(mask)
    for k in mapping_20:
        label_mask[mask == k] = mapping_20[k]
    return label_mask


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
    
def compute_iou(model, testloader, args):
    model = model.eval()
   
    union = torch.zeros(args.num_classes, 1,dtype=torch.float).cuda().float()
    inter = torch.zeros(args.num_classes, 1, dtype=torch.float).cuda().float()
    preds = torch.zeros(args.num_classes, 1, dtype=torch.float).cuda().float() 
    # extra 

    with torch.no_grad():
        for index, batch in tqdm(enumerate(testloader)):
            # print('******************') 
            image , seg_label, _ = batch  ## chg    
            output =  model(image.float().cuda())       

            label = seg_label.cuda()
            output = output.squeeze()  
            C,H,W = output.shape   

            #########################################################################
            Mask = (label.squeeze())<C  # it is ignoring all the labels values equal or greater than 19 #(1080, 1920) 
            pred_e = torch.linspace(0,C-1, steps=C).view(C, 1, 1)  
            pred_e = pred_e.repeat(1, H, W).cuda()  
            pred = output.argmax(dim=0).float() ## prior  
            pred_mask = torch.eq(pred_e, pred).byte()    
            pred_mask = pred_mask*Mask 
        
            label_e = torch.linspace(0,C-1, steps=C).view(C, 1, 1)
            label_e = label_e.repeat(1, H, W).cuda()
            label = label.view(1, H, W)
            label_mask = torch.eq(label_e, label.float()).byte()
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
            ##########################################################################


        iou = inter/union
        acc = inter/preds 
        iou = np.array(iou.cpu())
        iou = torch.tensor(np.where(np.isnan(iou), 0, iou))
        acc = np.array(acc.cpu())
        acc = torch.tensor(np.where(np.isnan(acc), 0, acc))

        mIoU = iou.mean().item()   
        mAcc = acc.mean().item() 
        print('====cityscapes pred====')
        print_iou(iou, acc, mIoU, mAcc)  ## original

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
        iou, mIoU, acc, mAcc = compute_iou(self.model, testloader, self.args)
        
        print('loss calc and saving images')
        ## not calculating loss for now 
        for i_iter, batch in tqdm(enumerate(testloader)): 
             
            image, seg_label, name = batch
            nm = name[0].split('/')[-1]
                        
            if not os.path.exists(os.path.join('city_pred', self.args.save_path)):
                os.makedirs(os.path.join('city_pred', self.args.save_path))  
            ## perturbation 
            with torch.no_grad():
                # write the 
                output =  self.model(image.float().cuda())     
                # output = output.squeeze() 
                output = torch.tensor(output)

                seg_preds = torch.argmax(output, dim=1) 
                seg_preds = [torch.tensor(label_img_to_color(seg_preds[sam], os.path.join('city_pred', self.args.save_path) + '/' + nm)) for sam in range(output.shape[0])]                 

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
        self.interp = nn.Upsample(size=(1080, 1920), mode='bilinear', align_corners=True)

    def eval(self):
        print('Lets eval...') 
        valid_loss = self.validate()    
        return     
 
def main(args): 
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

