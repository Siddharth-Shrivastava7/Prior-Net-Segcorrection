import torch  
import torch.multiprocessing as mp

from torch.utils import data 
import torchvision.transforms as transforms  
import os, sys 
import torch.backends.cudnn as cudnn 
from model.Unet import * 
from model.unet_resnetenc import *
from init_config import * 
from easydict import EasyDict as edict 
import numpy as np  
import random  
import time, datetime  
import torch.nn as nn 
from tqdm import tqdm 
import torch.nn.functional as F
from dataset.network.dannet_pred import pred    
from utils.optimize import * 
import  torch.optim as optim 
from PIL import Image  
from tqdm import tqdm 


## dataset init 
def init_train_data(cfg): 
    train_env = cfg[cfg.train] 
    cfg.train_data_dir = train_env.data_dir
    cfg.train_data_list = train_env.data_list
    cfg.input_size = train_env.input_size  

    trainloader = data.DataLoader(
            BaseDataSet(cfg, cfg.train_data_dir, cfg.train_data_list, cfg.train, cfg.num_class, ignore_label=255, set='train'),
            batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.worker, pin_memory=True)

    return trainloader,cfg

def init_val_data(cfg): 
    val_env = cfg[cfg.val] 
    cfg.val_data_dir = val_env.data_dir
    cfg.val_data_list = val_env.data_list
    cfg.input_size = val_env.input_size  

    valloader = data.DataLoader(
            BaseDataSet(cfg, cfg.val_data_dir, cfg.val_data_list, cfg.val, cfg.num_class, ignore_label=255, set='val'),
            batch_size=1, shuffle=True, num_workers=cfg.worker, pin_memory=True)

    return valloader
    

class BaseDataSet(data.Dataset): 
    def __init__(self, cfg, root, list_path,dataset, num_class, ignore_label=255, set='val'): 

        self.root = root 
        self.list_path = list_path
        self.set = set
        self.dataset = dataset 
        self.num_class = num_class 
        self.ignore_label = ignore_label
        self.cfg = cfg  
        self.img_ids = []   
        self.files = []

        if self.dataset == 'acdc_train_label' or self.dataset == 'acdc_val_label':  
            
            with open(self.list_path) as f: 
                for item in f.readlines(): 
                    fields = item.strip().split('\t')[0]
                    if ' ' in fields:
                        fields = fields.split(' ')[0]
                    self.img_ids.append(fields)  
          
            for name in self.img_ids: 
                img_file = osp.join(self.root, name) 
                replace = (("_rgb_anon", "_gt_labelTrainIds"), ("acdc_trainval", "acdc_gt"), ("rgb_anon", "gt"))
                nm = name
                for r in replace: 
                    nm = nm.replace(*r) 
                label_file = osp.join(self.root, nm)   
                self.files.append({
                        "img": img_file,
                        "label":label_file,
                        "name": name
                    })  

        elif self.dataset == 'perturb_bdd_city':  
            
            ## texture perturb bdd dataset read 
            with open(self.list_path[0]) as f: 
                for item in f.readlines(): 
                    fields = item.strip().split('\t')[0]
                    self.img_ids.append(fields)  
            
            ## texture perturb cityscapes dataset read 
            with open(self.list_path[1]) as f: 
                for item in f.readlines(): 
                    fields = item.strip().split('\t')[0]
                    self.img_ids.append(fields)  
            
            for name in self.img_ids: 
                if name.find('.jpg')!=-1: 
                    img_file = osp.join(self.root[0], name)
                    label_name = name.split('.')[0] + '_train_id.png' 
                    label_root = self.root[0].replace('texture_variant/bdd_acdc', 'bdd100k_seg/bdd100k/seg/labels/train')
                    label_file = osp.join(label_root, label_name)   
                
                else: 
                    name_l = name.split('/')[-1]
                    img_file = osp.join(self.root[1], name_l)
                    label_name = name.replace('leftImg8bit', 'gtFine_labelIds') 
                    label_root = self.root[1].replace("texture_variant/cityscapes_acdc", "cityscapes")
                    label_file = osp.join(label_root, 'gtFine/train',  label_name)

                self.files.append({
                        "img": img_file,
                        "label":label_file,
                        "name": name
                    })  

        elif self.dataset == 'acdc_bdd_city':  

            ## acdc read txt 
            with open(self.list_path[0]) as f: 
                for item in f.readlines(): 
                    fields = item.strip().split('\t')[0]
                    self.img_ids.append(fields)  
            
            ## bdd read txt
            with open(self.list_path[1]) as f: 
                for item in f.readlines(): 
                    fields = item.strip().split('\t')[0]
                    self.img_ids.append(fields) 
             
            ## city read txt 
            with open(self.list_path[2]) as f: 
                for item in f.readlines(): 
                    fields = item.strip().split('\t')[0]
                    self.img_ids.append(fields)

            for name in self.img_ids:  
                ## bdd
                if name.find('.jpg')!=-1: 
                    img_file = osp.join(self.root[1], name)
                    label_name = name.split('.')[0] + '_train_id.png' 
                    label_root = self.root[1].replace('images', 'labels')
                    label_file = osp.join(label_root, label_name) 
                
                ## city 
                elif len(name.split('/')) == 2:
                    img_file = osp.join(self.root[2] ,'leftImg8bit/train', name)
                    label_name = name.replace('leftImg8bit', 'gtFine_labelIds') 
                    label_file = osp.join(self.root[2], 'gtFine/train',  label_name)
            
                ## acdc 
                else: 
                    img_file = osp.join(self.root[0], name) 
                    replace = (("_rgb_anon", "_gt_labelTrainIds"), ("acdc_trainval", "acdc_gt"), ("rgb_anon", "gt"))
                    nm = name
                    for r in replace: 
                        nm = nm.replace(*r) 
                    label_file = osp.join(self.root[0], nm) 
                
                self.files.append({
                            "img": img_file,
                            "label":label_file,
                            "name": name
                        }) 

        elif self.dataset == 'bdd_city':  

            with open(self.list_path[0]) as f: 
                for item in f.readlines(): 
                    fields = item.strip().split('\t')[0]
                    self.img_ids.append(fields)  
            
            with open(self.list_path[1]) as f: 
                for item in f.readlines(): 
                    fields = item.strip().split('\t')[0]
                    self.img_ids.append(fields) 
            

            for name in self.img_ids:  
                ## bdd
                if name.find('.jpg')!=-1: 
                    img_file = osp.join(self.root[0], name)
                    label_name = name.split('.')[0] + '_train_id.png' 
                    label_root = self.root[0].replace('images', 'labels')
                    label_file = osp.join(label_root, label_name) 
                
                ## city 
                else:
                    img_file = osp.join(self.root[1] ,'leftImg8bit/train', name)
                    label_name = name.replace('leftImg8bit', 'gtFine_labelIds') 
                    label_file = osp.join(self.root[1], 'gtFine/train',  label_name)
        
                self.files.append({
                            "img": img_file,
                            "label":label_file,
                            "name": name
                        }) 

        
        elif self.dataset == 'bdd_city_trval':   

            with open(self.list_path[0] + 'bdd_' + self.set + '.txt') as f: 
                for item in f.readlines(): 
                    fields = item.strip().split('\t')[0]
                    self.img_ids.append(fields)  
            
            
            with open(self.list_path[1] + self.set + '.txt') as f: 
                for item in f.readlines(): 
                    fields = item.strip().split('\t')[0]
                    self.img_ids.append(fields) 


            for name in self.img_ids:  
                ## bdd 
                if name.find('.jpg')!=-1: 
                    img_file = osp.join(self.root[0], self.set, name)
                    label_name = name.split('.')[0] + '_train_id.png' 
                    label_root = self.root[0].replace('images', 'labels')
                    label_file = osp.join(label_root, self.set, label_name)   

                ## city   
                else:
                    img_file = osp.join(self.root[1] ,'leftImg8bit', self.set, name)
                    label_name = name.replace('leftImg8bit', 'gtFine_labelIds') 
                    label_file = osp.join(self.root[1], 'gtFine', self.set, label_name) 
        
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
            if self.dataset in ['acdc_train_label', 'acdc_val_label']: 
                mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  ## image net mean and std  
                
                if self.set=='val':  
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
                    
                else: 
                    image = Image.open(datafiles["img"]).convert('RGB') 

                    transforms_compose_img = transforms.Compose([transforms.Resize((512,512)), transforms.ToTensor(), transforms.Normalize(*mean_std)]) 
                    img_trans = transforms_compose_img(image) 
                    image = torch.tensor(np.array(img_trans)).float() 
                    
                    transforms_compose_label = transforms.Compose([transforms.Resize((512,512),interpolation=Image.NEAREST)]) 
                    label = Image.open(datafiles["label"]) 
                    label = transforms_compose_label(label)   
                    label = torch.tensor(np.array(label))   

            elif self.dataset in ['perturb_bdd_city', 'acdc_bdd_city', 'bdd_city', 'bdd_city_trval']:  
                mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  ## image net mean and std    

                if self.set=='val':   
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

                else: 
                    image = Image.open(datafiles["img"]).convert('RGB') 

                    transforms_compose_img = transforms.Compose([transforms.Resize((512,512)), transforms.ToTensor(), transforms.Normalize(*mean_std)])  
                    img_trans = transforms_compose_img(image) 
                    image = torch.tensor(np.array(img_trans)).float() 
                    
                    transforms_compose_label = transforms.Compose([transforms.Resize((512,512),interpolation=Image.NEAREST)]) 
                    label = Image.open(datafiles["label"]) 
                    label = transforms_compose_label(label)   
                    label = torch.tensor(np.array(label))   
            
            
        except: 
            # print('**************') 
            # print(index)
            index = index - 1 if index > 0 else index + 1 
            return self.__getitem__(index) 

        return image, label, name

                

def init_model(cfg):
    # model = UNet_mod(cfg.num_channels, cfg.num_class, cfg.small).cuda()    
    
    model = UNetWithResnet50Encoder(cfg.num_ip_channels, cfg.num_class)
       
    params = torch.load(cfg.eval_restore_from)
    model.load_state_dict(params)
    print('----------Model initialize with weights from-------------: {}'.format(cfg.eval_restore_from))
    if cfg.multigpu:
        model = nn.DataParallel(model)  
    model.eval().cuda()
    print('Mode --> Eval') 
    return model 
    

def print_iou(iou, acc, miou, macc):
    for ind_class in range(iou.shape[0]):
        print('===> {0:2d} : {1:.2%} {2:.2%}'.format(ind_class, iou[ind_class, 0].item(), acc[ind_class, 0].item()))
    print('mIoU: {:.2%} mAcc : {:.2%} '.format(miou, macc))


def compute_iou(model, testloader, cfg, da_model, lightnet, weights):
    model = model.eval()

    interp = nn.Upsample(size=(1080,1920), mode='bilinear', align_corners=True)   # dark_zurich -> (1080,1920) 
    union = torch.zeros(cfg.num_class, 1,dtype=torch.float).cuda().float()
    inter = torch.zeros(cfg.num_class, 1, dtype=torch.float).cuda().float()
    preds = torch.zeros(cfg.num_class, 1, dtype=torch.float).cuda().float()
    # extra 
    gts = torch.zeros(cfg.num_class, 1, dtype=torch.float).cuda().float() 

    union2 = torch.zeros(cfg.num_class, 1,dtype=torch.float).cuda().float()
    inter2 = torch.zeros(cfg.num_class, 1, dtype=torch.float).cuda().float()
    preds2 = torch.zeros(cfg.num_class, 1, dtype=torch.float).cuda().float()
    # extra 
    gts2 = torch.zeros(cfg.num_class, 1, dtype=torch.float).cuda().float() 

    with torch.no_grad():
        for index, batch in tqdm(enumerate(testloader)):
            # print('******************') 
            image , label, name = batch  ## chg 
            # print(label.shape) # torch.Size([1, 1080, 1920]) 
            
            interp = nn.Upsample(size=(1080, 1920), mode='bilinear', align_corners=True)
            # dannet prediction 
            da_model = da_model.eval()
            lightnet = lightnet.eval()

            r = lightnet(image.cuda())
            enhancement = image.cuda() + r
            if model == 'RefineNet':
                output2 = da_model(enhancement)
            else:
                _, output2 = da_model(enhancement)

            # if cfg.method_eval == 'shuffle_w_weights': 
            weights_prob = weights.expand(output2.size()[0], output2.size()[3], output2.size()[2], 19)
            weights_prob = weights_prob.transpose(1, 3)
            output2 = output2 * weights_prob

            output2 = interp(output2).cpu().numpy() 
            seg_tensor = torch.tensor(output2)   
            seg_tensor = F.softmax(seg_tensor, dim=1)  
            # print(seg_tensor.shape) # torch.Size([1, 19, 1080, 1920]) 
            # if cfg.error_inds:
            if cfg.method_eval == 'model_onehot':
                dannet_label = seg_tensor.argmax(dim=1).float().squeeze() ## prior   
                dannet_label[dannet_label==255] = 19   
                # print(dannet_label.unique()) # 0 to 19
                label_perturb_tensor = F.one_hot(dannet_label.to(torch.int64), 20) 
                label_perturb_tensor = label_perturb_tensor[:, :, :19]  
                # print(label_perturb_tensor.shape) # torch.Size([1080, 1920, 19])   
                # print(torch.unique(label_perturb_tensor)) # 0,1
                label_perturb_tensor = label_perturb_tensor.unsqueeze(dim=0).transpose(3,2).transpose(2,1)
                # print(label_perturb_tensor.shape) # torch.Size([1, 19, 1080, 1920]) 
                # print(dannet_label.shape) # torch.Size([1, 1080, 1920]) 
                # dannet_label = dannet_label.squeeze() 
                # label = label.squeeze() 
                # print(label.shape, dannet_label.shape)  
                # error = (label != dannet_label).nonzero(as_tuple=True)
                # print(torch.unique(error))
                # print(len(error)) # 2 
                

            if cfg.method_eval=='shuffle' or cfg.method_eval == 'shuffle_w_weights': 
                randx_pix = np.random.randint(seg_tensor.shape[2]-10)
                randy_pix = np.random.randint(seg_tensor.shape[2]-10)
                # print(randx_pix, randy_pix) 
            
                for x in range(10):
                    for y in range(10):  
                        # print('>>>>>>>>>>>>')
                        org_batch_val = seg_tensor[:, :, randx_pix + x, randy_pix + y]  
                        shuffle_inds = torch.randperm(org_batch_val.shape[1]) 
                        # print(shuffle_inds)   
                        seg_tensor[:, :, randx_pix + x, randy_pix + y] = org_batch_val[:, shuffle_inds] 
        
                label_perturb_tensor = seg_tensor.detach().clone()  
            
            if cfg.method_eval=='straight':
                label_perturb_tensor = seg_tensor.detach().clone()  

            output =  model(label_perturb_tensor.float().cuda())       
            perturb = label_perturb_tensor.cuda() 
            label = label.cuda()
            output = output.squeeze()  
            perturb = perturb.squeeze()
            C,H,W = output.shape   
         
            #########################################################################corrected one 
            Mask = (label.squeeze())<C  # it is ignoring all the labels values equal or greater than 19 #(1080, 1920) 
            pred_e = torch.linspace(0,C-1, steps=C).view(C, 1, 1)  
            pred_e = pred_e.repeat(1, H, W).cuda()  
            pred = output.argmax(dim=0).float() ## prior  
            # print(pred.shape) # torch.Size([1080, 1920]) 
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
            ##########################################################################corrected one 

            #########################################################################perturbed one 
            pred = perturb.argmax(dim=0).float() ## perturb 
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
        iou = inter/union
        acc = inter/preds
        mIoU = iou.mean().item()
        mAcc = acc.mean().item()
        # print('*********')
        # print(gts)
        print_iou(iou, acc, mIoU, mAcc)  ## original
        ################## 

        ###########original
        iou2 = inter2/union2
        acc2 = inter2/preds2
        mIoU2 = iou2.mean().item()
        mAcc2 = acc2.mean().item()
        # print('*********')
        # print(gts)
        print_iou(iou2, acc2, mIoU2, mAcc2)  ## original
        ##################

        
        return iou, mIoU, acc, mAcc

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
        19: [0,  0, 0]
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


class BaseTrainer(object):
    def __init__(self, models, optimizers, loaders, config,  writer):
        self.model = models
        self.optim = optimizers
        self.loader = loaders
        self.config = config
        self.output = {}
        self.writer = writer
        
    def validate(self): 
        self.model = self.model.eval() 
        total_loss = 0
        testloader = init_val_data(self.config) 
        print('In validation') 
        
        if self.config.rgb_save:  

            ## not calculating loss for now 
            for i_iter, batch in tqdm(enumerate(testloader)):
                image, seg_label, name = batch 
                nm = name[0].split('/')[-1] 

                ## dannet prediction which will further act as an input to the prior net  
                with torch.no_grad():
                    r = self.lightnet(image.cuda())  
                    enhancement = image.cuda() + r  
                    if self.config.model_dannet == 'RefineNet':
                        output2 = self.da_model(enhancement)
                    else:
                        _, output2 = self.da_model(enhancement)

                ## weighted cross entropy for which they have been used so here not using it in the training so ignoring for now 
                weights_prob = self.weights.expand(output2.size()[0], output2.size()[3], output2.size()[2], 19)
                weights_prob = weights_prob.transpose(1, 3) 
                output2 = output2 * weights_prob   

                # print(output2.shape)  # torch.Size([1, 19, 68, 121]) 
                # print('>>>>>>>>>>')

                output2 = self.interp(output2).cpu().numpy() 
                seg_pred_tensor = torch.tensor(output2) 
                # print(seg_pred_tensor.shape) # torch.Size([1, 19, 1080, 1920]) 
                # print('<<<<<<<')

                seg_correct = self.model(seg_pred_tensor.float().cuda())   
                # print(seg_correct.shape) # torch.Size([1, 19, 1080, 1920])


                if not os.path.exists(os.path.join('save_pred', self.config.save_path)):
                    os.makedirs(os.path.join('save_pred', self.config.save_path))  
                if not os.path.exists(os.path.join('save_cor', self.config.save_path)):
                    os.makedirs(os.path.join('save_cor', self.config.save_path)) 

                seg_preds = torch.argmax(seg_pred_tensor, dim=1) 
                seg_preds = [torch.tensor(label_img_to_color(seg_preds[sam], os.path.join('save_pred', self.config.save_path) + '/' + nm)) for sam in range(seg_pred_tensor.shape[0])]  

                seg_preds = torch.argmax(seg_correct, dim=1) 
                seg_preds = [torch.tensor(label_img_to_color(seg_preds[sam], os.path.join('save_cor', self.config.save_path) + '/' + nm)) for sam in range(seg_pred_tensor.shape[0])] 

                seg_label = seg_label.long().cuda() # cross entropy     
                loss = CrossEntropy2d() # ce loss  
                seg_loss = loss(seg_correct, seg_label) 
                total_loss += seg_loss.item()   

            total_loss /= len(iter(testloader))
            print('---------------------')
            print('Validation seg loss: {}'.format(total_loss))   
        

        print("MIou calculation: ")    
        iou, mIoU, acc, mAcc = compute_iou(self.model, testloader, self.config, self.da_model, self.lightnet, self.weights)  

        return total_loss

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

## extra for one hot encoding
# def get_one_hot( label , N ): 
    # size = list(label.size()) 
    # label = label.view(-1) 
    # ones = torch.sparse.torch.eye( N ) 
    # ones = ones.index_select(0, label) 
    # size.append(N)
    # return ones.view(*size)


class Trainer(BaseTrainer):
    def __init__(self, model, config, writer):
        self.model = model
        self.config = config
        self.writer = writer
        self.da_model, self.lightnet, self.weights = pred(self.config.num_class, self.config.model_dannet, self.config.restore_from_da, self.config.restore_light_path)
        self.interp = nn.Upsample(size=(1080, 1920), mode='bilinear', align_corners=True)

    def eval(self):
        
        if self.config.multigpu:       
            self.optim = optim.SGD(self.model.module.optim_parameters(self.config.learning_rate),
                          lr=self.config.learning_rate, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        else:
            self.optim = optim.SGD(self.model.parameters(),
                          lr=self.config.learning_rate, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
            # self.optim = optim.Adam(self.model.parameters(),
            #             lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
         
        print('Lets eval...') 
        valid_loss = self.validate()    
        return     
 

def main(): 
    os.environ['CUDA_VISIBLE_DEVICES'] = '5' 
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    cudnn.enabled = True
    cudnn.benchmark = True   
    config, writer = init_config("config/config_exp.yml", sys.argv)
    model = init_model(config)
    trainer = Trainer(model, config, writer)
    trainer.eval() 


if __name__ == "__main__": 
    mp.set_start_method('spawn')   ## for different random value using np.random 
    start = datetime.datetime(2020, 1, 22, 23, 00, 0)
    print("wait")
    while datetime.datetime.now() < start:
        time.sleep(1)
    main()

        

        
        
        
        
         
        
        
        
        
        
            
         

        
        
        
        

        
        
        
            
            
            
            




        


        

    


            

    
    
    

    
    
    
    
    
        

    
    
    
    
    