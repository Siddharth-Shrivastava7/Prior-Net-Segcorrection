import torch  
import torch.multiprocessing as mp

from torch.utils import data 
import torchvision.transforms as transforms  
import os, sys 
import torch.backends.cudnn as cudnn 
from model.Unet import *  
from model.unet_model import * 
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
from sklearn.feature_extraction.image import extract_patches_2d


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

        with open(self.list_path) as f: 
            for item in f.readlines(): 
                fields = item.strip().split('\t')[0]
                if ' ' in fields:
                    fields = fields.split(' ')[0]
                self.img_ids.append(fields)  
        
        self.files = []  
        if self.dataset == 'acdc_train_label' or self.dataset == 'acdc_val_label':  
            
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

    def __len__(self):
        return len(self.files) 
    
    def __getitem__(self, index): 
        datafiles = self.files[index]
        name = datafiles["name"]    
        
        try:  
            if self.dataset in ['acdc_val_label']: 

                mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  ## image net mean and std   
                transforms_compose_label = transforms.Compose([
                            transforms.Resize((1080,1920), interpolation=Image.NEAREST)])  

                label = Image.open(datafiles["label"])  
                label = transforms_compose_label(label)

                label_perturb = np.array(label)   

                if self.cfg.method_eval=='perturb': 
                    ## random discrete discontinuous pixels 
                    for _ in range(self.cfg.num_pix_perturb): 
                        randx, randy = np.random.randint(label_perturb.shape[0]), np.random.randint(label_perturb.shape[1])   
                        actual_label = label_perturb[randx, randy]
                        while actual_label!=255:
                            perturb_label = np.random.randint(19)  
                            if actual_label!= perturb_label: break  
                        if actual_label == 255: 
                            perturb_label = 255 
                        label_perturb[randx,randy] = perturb_label
                    
                    label = torch.tensor(np.array(label))  

                    label_perturb = Image.fromarray(label_perturb)    
                    label_perturb_tensor = torch.tensor(np.array(label_perturb))  
                    label_perturb_tensor[label_perturb_tensor==255] = 19  
                    label_perturb_tensor = F.one_hot(label_perturb_tensor.to(torch.int64), 20) 
                    label_perturb_tensor = label_perturb_tensor[:, :, :19] 

                elif self.cfg.method_eval=='perturb_discrete_patches': 
                    ## random discrete discontinuous patches 
                    ## square patch
                    for _ in range(self.cfg.num_patch_perturb):  
                        randy, randx = np.random.randint(label_perturb.shape[0] - self.cfg.patch_size), np.random.randint(label_perturb.shape[1] - self.cfg.patch_size)    

                        lp_patch = label_perturb[randy:randy+ self.cfg.patch_size, randx:randx+ self.cfg.patch_size]  
                        
                        for y in range(lp_patch.shape[0]): 
                            for x in range(lp_patch.shape[1]):
                                actual_label = lp_patch[y,x] 
                                while actual_label!=255:
                                    perturb_label = np.random.randint(19)  
                                    if actual_label!= perturb_label: break  
                                if actual_label == 255: 
                                    perturb_label = 255 
                                lp_patch[y,x] = perturb_label
                        
                        label_perturb[randy:randy+ self.cfg.patch_size, randx:randx+ self.cfg.patch_size] = lp_patch 
                    
                    
                    label = torch.tensor(np.array(label))  

                    label_perturb = Image.fromarray(label_perturb)    
                    label_perturb_tensor = torch.tensor(np.array(label_perturb))  
                    label_perturb_tensor[label_perturb_tensor==255] = 19  
                    label_perturb_tensor = F.one_hot(label_perturb_tensor.to(torch.int64), 20) 
                    label_perturb_tensor = label_perturb_tensor[:, :, :19]            
                
                elif self.cfg.method_eval=='one_pix': 
                
                    ## random single pixel 
                    rand_x = np.random.randint(label_perturb.shape[0])  
                    rand_y = np.random.randint(label_perturb.shape[1])    
                    # print(rand_x, rand_y)

                    """for x in range(10): 
                        for y in range(10): 
                            actual_label = label_perturb[rand_x+x, rand_y+y] 
                            while True and actual_label!=255:   
                                perturb_label = np.random.randint(19)  
                                if actual_label!= perturb_label: break 
                            if actual_label == 255: 
                                perturb_label = 255 
                            label_perturb[rand_x+x, rand_y+y]  = perturb_label""" 
                    
                    actual_label = label_perturb[rand_x, rand_y] 
                    while True and actual_label!=255: 
                        perturb_label = np.random.randint(19)   
                        if actual_label!= perturb_label: break  
                    if actual_label == 255: 
                        perturb_label = 255 
                    label_perturb[rand_x, rand_y]  = perturb_label 

                    label_perturb = Image.fromarray(label_perturb)
                    label_perturb = transforms_compose_label(label_perturb)      
                    label_perturb_tensor = torch.tensor(np.array(label_perturb))  
                    label_perturb_tensor[label_perturb_tensor==255] = 19  
                    label_perturb_tensor = F.one_hot(label_perturb_tensor.to(torch.int64), 20) 
                    label_perturb_tensor = label_perturb_tensor[:, :, :19]  

                elif self.cfg.method_eval == 'patch_pix': 

                    rand_x = np.random.randint(label_perturb.shape[0]-10)  
                    rand_y = np.random.randint(label_perturb.shape[1]-10)   
                    # print(rand_x, rand_y)

                    for x in range(10): 
                        for y in range(10): 
                            actual_label = label_perturb[rand_x+x, rand_y+y] 
                            while True and actual_label!=255:   
                                perturb_label = np.random.randint(19)  
                                if actual_label!= perturb_label: break 
                            if actual_label == 255: 
                                perturb_label = 255 
                            label_perturb[rand_x+x, rand_y+y]  = perturb_label
                    
                    label_perturb = Image.fromarray(label_perturb)
                    label_perturb = transforms_compose_label(label_perturb)      
                    label_perturb_tensor = torch.tensor(np.array(label_perturb))  
                    label_perturb_tensor[label_perturb_tensor==255] = 19  
                    label_perturb_tensor = F.one_hot(label_perturb_tensor.to(torch.int64), 20) 
                    label_perturb_tensor = label_perturb_tensor[:, :, :19] 

                elif self.cfg.method_eval == 'gt_correct': 
                    label_perturb = Image.fromarray(label_perturb)
                    label_perturb = transforms_compose_label(label_perturb)      
                    label_perturb_tensor = torch.tensor(np.array(label_perturb))  
                    label_perturb_tensor[label_perturb_tensor==255] = 19  
                    label_perturb_tensor = F.one_hot(label_perturb_tensor.to(torch.int64), 20) 
                    label_perturb_tensor = label_perturb_tensor[:, :, :19]   

                    label = torch.tensor(np.array(label))

                elif self.cfg.method_eval == 'patch_pix_large': 
                    rand_x = np.random.randint(label_perturb.shape[0]-500)  
                    rand_y = np.random.randint(label_perturb.shape[1]-500)   
                    # print(rand_x, rand_y) 

                    for x in range(500): 
                        for y in range(500): 
                            actual_label = label_perturb[rand_x+x, rand_y+y] 
                            while True and actual_label!=255:   
                                perturb_label = np.random.randint(19)  
                                if actual_label!= perturb_label: break 
                            if actual_label == 255: 
                                perturb_label = 255 
                            label_perturb[rand_x+x, rand_y+y]  = perturb_label
                    
                    label_perturb = Image.fromarray(label_perturb)
                    label_perturb = transforms_compose_label(label_perturb)      
                    label_perturb_tensor = torch.tensor(np.array(label_perturb))  
                    label_perturb_tensor[label_perturb_tensor==255] = 19  
                    label_perturb_tensor = F.one_hot(label_perturb_tensor.to(torch.int64), 20) 
                    label_perturb_tensor = label_perturb_tensor[:, :, :19]

                elif self.cfg.method_eval == 'singlech': 
                    ## random single pixel 
                    rand_x = np.random.randint(label_perturb.shape[0])  
                    rand_y = np.random.randint(label_perturb.shape[1])   
                    # print(rand_x, rand_y) 
                    
                    actual_label = label_perturb[rand_x, rand_y] 
                    while True and actual_label!=255: 
                        perturb_label = np.random.randint(19)   
                        if actual_label!= perturb_label: break  
                    if actual_label == 255: 
                        perturb_label = 255 
                    label_perturb[rand_x, rand_y]  = perturb_label 

                    label = transforms_compose_label(label) 
                    label = torch.tensor(np.array(label)) 
                    label_perturb = Image.fromarray(label_perturb)
                    label_perturb = transforms_compose_label(label_perturb)      
                    label_perturb_tensor = torch.tensor(np.array(label_perturb))  
                    label_perturb_tensor = label_perturb_tensor.unsqueeze(dim=2) ## one channel 

                elif self.cfg.method_eval == 'perturb_both':  

                    label = np.array(label)## saving the original
                    
                    for _ in range(self.cfg.num_pix_perturb): 
                        randy, randx = np.random.randint(label_perturb.shape[0]), np.random.randint(label_perturb.shape[1])   
                        actual_label = label[randy, randx]
                        while actual_label!=255: 
                            perturb_label = np.random.randint(19)  
                            if actual_label!= perturb_label: break  
                        if actual_label == 255: 
                            perturb_label = actual_label 
                        label_perturb[randy,randx] = perturb_label 

                    for _ in range(self.cfg.num_patch_perturb):  
                        randy, randx = np.random.randint(label_perturb.shape[0] - self.cfg.patch_size), np.random.randint(label_perturb.shape[1] - self.cfg.patch_size)    

                        lp_patch = label[randy:randy+ self.cfg.patch_size, randx:randx+ self.cfg.patch_size]  
                        
                        for y in range(lp_patch.shape[0]): 
                            for x in range(lp_patch.shape[1]):
                                actual_label = lp_patch[y,x] 
                                while actual_label!=255:
                                    perturb_label = np.random.randint(19)  
                                    if actual_label!= perturb_label: break  
                                if actual_label == 255: 
                                    perturb_label = 255 
                                lp_patch[y,x] = perturb_label
                        
                        label_perturb[randy:randy+ self.cfg.patch_size, randx:randx+ self.cfg.patch_size] = lp_patch
                    
                    label = torch.tensor(label)  

                    label_perturb = Image.fromarray(label_perturb)    
                    label_perturb_tensor = torch.tensor(np.array(label_perturb))  
                    label_perturb_tensor[label_perturb_tensor==255] = 19  
                    label_perturb_tensor = F.one_hot(label_perturb_tensor.to(torch.int64), 20) 
                    label_perturb_tensor = label_perturb_tensor[:, :, :19]   
                
                elif self.cfg.method_eval == 'model_onehot': 
                    ## image for transforming  
                    transforms_compose_img = transforms.Compose([
                        transforms.Resize((540, 960)),
                        transforms.ToTensor(),
                        transforms.Normalize(*mean_std), ## use only in training 
                    ])
                    image = Image.open(datafiles["img"]).convert('RGB') 
                    image = transforms_compose_img(image)    
                    image = torch.tensor(np.array(image)).float() 

                    label = torch.tensor(np.array(label))  

                    label_perturb_tensor = label.clone() 
                    ## will be doing later on 
                    # label_perturb_tensor[label_perturb_tensor==255] = 19  
                    # label_perturb_tensor = F.one_hot(label_perturb_tensor.to(torch.int64), 20) 
                    # label_perturb_tensor = label_perturb_tensor[:, :, :19] 

                    return image, label_perturb_tensor, label, name
                                    
        except: 
            print('**************') 
            print(index)
            index = index - 1 if index > 0 else index + 1 
            return self.__getitem__(index) 

        return label_perturb_tensor, label, name
                

def init_model(cfg):
    # unet = cfg.unet
    # model = UNet_mod(cfg.num_channels, cfg.num_class, cfg.small).cuda() ## previous unet model   
    # model = UNet(unet.enc_chs, unet.dec_chs, unet.num_class).cuda()    
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
            if cfg.method_eval == 'model_onehot' and cfg.model_pred == 'dannet_error_patch':  
                # print('>>>>>>>>>>>>>>>>>>')
                image, label_perturb_tensor, seg_label, name = batch 
                # label_perturb_tensor = label_perturb_tensor.transpose(3,2).transpose(2,1)
                ## dannet pred 

                with torch.no_grad():
                    r = lightnet(image.cuda())  
                    enhancement = image.cuda() + r  
                    if cfg.model_dannet == 'RefineNet':
                        output2 = da_model(enhancement)
                    else: 
                        _, output2 = da_model(enhancement)

                ## weighted cross entropy for which they have been used so here not using it in the training so ignoring for now 
                weights_prob = weights.expand(output2.size()[0], output2.size()[3], output2.size()[2], 19)
                weights_prob = weights_prob.transpose(1, 3) 
                output2 = output2 * weights_prob   
                     
                output2 = interp(output2).cpu().numpy() 
                seg_tensor = torch.tensor(output2)   
                seg_tensor = F.softmax(seg_tensor, dim=1)  
                dannet_label = seg_tensor.argmax(dim=1).float().squeeze()
                seg_label = seg_label.squeeze() 
                # print(dannet_label.shape) # torch.Size([1080, 1920])
                # print(seg_label.shape) # torch.Size([1080, 1920]) 
                ## error map calculation 
                # print(torch.unique(seg_label), torch.unique(dannet_label))
                dannet_label[seg_label == 255] = 255 
                error_inds = (seg_label != dannet_label).nonzero(as_tuple=False)  
                # print(error_inds.shape) 
                # print(error_inds) 
                # print(error_inds.shape[0]) 
                
                ## patch of error transfer
                random_num_list = random.sample([i for i in range(error_inds.shape[0])], cfg.num_pix_perturb) 
                # print(random_num_list) 
                # print(len(random_num_list))
                # print('*********') 
                # whole error transfer
                # random_num_list = random.sample([i for i in range(error_inds.shape[0])], error_inds.shape[0])

                # print(error_inds[random_num_list].shape) 
                error_discrete_patch = error_inds[random_num_list]  
                # print(error_discrete_patch)  
                # print(error_discrete_patch[:,0])
                # print(dannet_label[error_discrete_patch[:,0], error_discrete_patch[:,1]].shape)  

                ## patch error transfer 
                dannet_error_patch = dannet_label[error_discrete_patch[:,0], error_discrete_patch[:,1]] 

                # print(dannet_error_patch.unique()) 
                # print(dannet_error_patch) 
                # print(dannet_error_patch.cpu().detach().numpy())
                # print(label_perturb_tensor.shape)  # torch.Size([1, 1080, 1920])
                label_perturb_tensor = label_perturb_tensor.squeeze() 
                # print(label_perturb_tensor[error_discrete_patch[:,0], error_discrete_patch[:,1]].shape) 
                # print(dannet_error_patch.byte().unique())
                label_perturb_tensor[error_discrete_patch[:,0], error_discrete_patch[:,1]] = dannet_error_patch.byte()

                ## converting perturbed label into 19 dim tensor 
                # print(label_perturb_tensor.shape)         
                # print(torch.unique(label_perturb_tensor)) 
                label_perturb_tensor[label_perturb_tensor==255] = 19   
                label_perturb_tensor = F.one_hot(label_perturb_tensor.to(torch.int64), 20) 
                label_perturb_tensor = label_perturb_tensor[:, :, :19] 
                # print(label_perturb_tensor.shape)  # torch.Size([1080, 1920, 19]) 
                label_perturb_tensor = label_perturb_tensor.unsqueeze(dim=0)
                label_perturb_tensor = label_perturb_tensor.transpose(3,2).transpose(2,1) 
                output = model(label_perturb_tensor.float().cuda())
                
            else: 
                # print('******************') 
                label_perturb_tensor, seg_label, name = batch  ## chg 
                
                # interp = nn.Upsample(size=(1080, 1920), mode='bilinear', align_corners=True)
                ## dannet prediction 
                # da_model = da_model.eval()
                # lightnet = lightnet.eval()

                # r = lightnet(image.cuda())
                # enhancement = image.cuda() + r
                # if model == 'RefineNet':
                #     output2 = da_model(enhancement)
                # else:
                #     _, output2 = da_model(enhancement)

                # weights_prob = weights.expand(output2.size()[0], output2.size()[3], output2.size()[2], 19)
                # weights_prob = weights_prob.transpose(1, 3)
                # output2 = output2 * weights_prob
                # output2 = interp(output2).cpu().numpy() 
                # image = torch.tensor(output2) 
                
                label_perturb_tensor = label_perturb_tensor.transpose(3,2).transpose(2,1) 
                output =  model(label_perturb_tensor.float().cuda())  

    
            perturb = label_perturb_tensor.cuda() 
            label = seg_label.cuda()
            output = output.squeeze()  
            perturb = perturb.squeeze()
            C,H,W = output.shape   
         
            #########################################################################prior net  
            Mask = (label.squeeze())<C  # it is ignoring all the labels values equal or greater than 2 #(1080, 1920) 
            pred_e = torch.linspace(0,C-1, steps=C).view(C, 1, 1)  
            pred_e = pred_e.repeat(1, H, W).cuda()  
            
            ## output model correction ## prior 
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
            # print(tmp_inter.shape)  # torch.Size([19, 1080, 1920])
            cu_inter = (tmp_inter==2).view(C, -1).sum(dim=1, keepdim=True).float()
            cu_union = (tmp_inter>0).view(C, -1).sum(dim=1, keepdim=True).float()
            cu_preds = pred_mask.view(C, -1).sum(dim=1, keepdim=True).float() 
            # extra
            # cu_gts = label_mask.view(C, -1).sum(dim=1, keepdim=True).float()
            # gts += cu_gts
            union+=cu_union
            inter+=cu_inter
            preds+=cu_preds
            #########################################################################prior net 


            #########################################################################perturbed
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
            #########################################################################perturbed


        ###########original
        iou = inter/union
        acc = inter/preds 
        # iou = torch.nan_to_num(iou) 
        # acc = torch.nan_to_num(acc)  
        iou = np.array(iou.cpu())
        iou = torch.tensor(np.where(np.isnan(iou), 0, iou))
        acc = np.array(acc.cpu())
        acc = torch.tensor(np.where(np.isnan(acc), 0, acc))

        mIoU = iou.mean().item()   
        mAcc = acc.mean().item() 
        # print('*********')
        # print(gts)
        print('====Prior Net====')
        print_iou(iou, acc, mIoU, mAcc)  ## original
        ################## 

        ###########original ##perturbed
        iou2 = inter2/union2
        acc2 = inter2/preds2
        iou2 = np.array(iou2.cpu())
        iou2 = torch.tensor(np.where(np.isnan(iou2), 0, iou2))
        acc2 = np.array(acc2.cpu())
        acc2 = torch.tensor(np.where(np.isnan(acc2), 0, acc2))

        mIoU2 = iou2.mean().item()
        mAcc2 = acc2.mean().item() 
        # print('*********')
        # print(gts)
        print('====Perturbed====')
        print_iou(iou2, acc2, mIoU2, mAcc2)  ## original
        ##################
        
        return iou, mIoU, acc, mAcc


class BaseTrainer(object):
    def __init__(self, models, optimizers, loaders, config,  writer):
        self.model = models
        self.optim = optimizers
        self.loader = loaders
        self.config = config
        self.output = {}
        self.writer = writer
        self.da_model, self.lightnet, self.weights = pred(self.config.num_class, self.config.model_dannet, self.config.restore_from_da, self.config.restore_light_path) 

    def validate(self): 
        self.model = self.model.eval() 
        total_loss = 0
        testloader = init_val_data(self.config) 

        if self.config.model_pred == 'dannet_error_patch' and self.config.rgb_save: 
            ## not calculating loss for now  
            for i_iter, batch in tqdm(enumerate(testloader)):
                image, label_perturb_tensor, seg_label, name = batch 
                # label_perturb_tensor = label_perturb_tensor.transpose(3,2).transpose(2,1)
                ## dannet pred 
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
                     
                output2 = self.interp(output2).cpu().numpy() 
                seg_tensor = torch.tensor(output2)   
                seg_tensor = F.softmax(seg_tensor, dim=1)  
                dannet_label = seg_tensor.argmax(dim=1).float().squeeze()
                seg_label = seg_label.squeeze() 
                # print(dannet_label.shape) # torch.Size([1080, 1920])
                # print(seg_label.shape) # torch.Size([1080, 1920]) 
                ## error map calculation 
                # print(torch.unique(seg_label), torch.unique(dannet_label))
                dannet_label[seg_label == 255] = 255 
                error_inds = (seg_label != dannet_label).nonzero(as_tuple=False)  
                # print(error_inds.shape) 
                # print(error_inds) 
                # print(error_inds.shape[0]) 
                
                ## patch of error transfer
                random_num_list = random.sample([i for i in range(error_inds.shape[0])], self.config.num_pix_perturb) 
                # print(random_num_list) 
                # print(len(random_num_list))
                # print('*********') 
                # whole error transfer
                # random_num_list = random.sample([i for i in range(error_inds.shape[0])], error_inds.shape[0])

                # print(error_inds[random_num_list].shape) 
                error_discrete_patch = error_inds[random_num_list]  
                # print(error_discrete_patch)  
                # print(error_discrete_patch[:,0])
                # print(dannet_label[error_discrete_patch[:,0], error_discrete_patch[:,1]].shape)  

                ## patch error transfer 
                dannet_error_patch = dannet_label[error_discrete_patch[:,0], error_discrete_patch[:,1]]

                # print(dannet_error_patch.unique()) 
                # print(dannet_error_patch) 
                # print(dannet_error_patch.cpu().detach().numpy())
                # print(label_perturb_tensor.shape)  # torch.Size([1, 1080, 1920])
                label_perturb_tensor = label_perturb_tensor.squeeze() 
                # print(label_perturb_tensor[error_discrete_patch[:,0], error_discrete_patch[:,1]].shape) 
                # print(dannet_error_patch.byte().unique())
                label_perturb_tensor[error_discrete_patch[:,0], error_discrete_patch[:,1]] = dannet_error_patch.byte()

                ## converting perturbed label into 19 dim tensor 
                # print(label_perturb_tensor.shape)         
                # print(torch.unique(label_perturb_tensor)) 
                label_perturb_tensor[label_perturb_tensor==255] = 19 
                label_perturb_tensor = F.one_hot(label_perturb_tensor.to(torch.int64), 20) 
                label_perturb_tensor = label_perturb_tensor[:, :, :19] 
                # print(label_perturb_tensor.shape)  # torch.Size([1080, 1920, 19]) 
                label_perturb_tensor = label_perturb_tensor.unsqueeze(dim=0)
                label_perturb_tensor = label_perturb_tensor.transpose(3,2).transpose(2,1) 
                output = self.model(label_perturb_tensor.float().cuda())  
                # print(seg_pred.shape)  # torch.Size([1, 19, 1080, 1920])  
                ## saving images 
                # print(seg_pred.shape)            

                # print(output.shape, label_perturb_tensor.shape) # torch.Size([1, 19, 1080, 1920]) torch.Size([1, 19, 1080, 1920])
                seg_preds = torch.argmax(label_perturb_tensor, dim=1) 
                # print(seg_label.shape, seg_preds.shape)  # torch.Size([1080, 1920]) torch.Size([1, 1080, 1920]) 
                # seg_preds = seg_preds.squeeze() 
                
                seg_preds[0][seg_label == 255] = 19  # since bt = 1  
                nm = name[0].split('/')[-1]  

                # print(seg_label.shape, seg_preds.shape) # torch.Size([1080, 1920]) torch.Size([1080, 1920])

                if not os.path.exists(os.path.join('save_pred', self.config.save_path)):
                    os.makedirs(os.path.join('save_pred', self.config.save_path))  
                if not os.path.exists(os.path.join('save_cor', self.config.save_path)):
                    os.makedirs(os.path.join('save_cor', self.config.save_path)) 
                seg_preds = [torch.tensor(label_img_to_color(seg_preds[sam], os.path.join('save_pred', self.config.save_path) + '/' + nm)) for sam in range(seg_preds.shape[0])]  
                output = torch.argmax(output, dim=1)   
                output[0][seg_label == 255] = 19  # since bt = 1
                output = [torch.tensor(label_img_to_color(output[sam], os.path.join('save_cor', self.config.save_path) + '/' + nm)) for sam in range(output.shape[0])]   
                
                # seg_preds = torch.tensor(label_img_to_color(seg_preds, os.path.join('save_pred', self.config.save_path) + '/' + nm)) 
                # seg_preds = torch.argmax(seg_preds, dim=1)   
                # seg_preds[seg_label == 255] = 19 
                # seg_preds = torch.tensor(label_img_to_color(seg_preds, os.path.join('save_cor', self.config.save_path) + '/' + nm)) 


            #     seg_label = seg_label.long().cuda() # cross entropy  
            #     loss = CrossEntropy2d() # ce loss  
            #     seg_loss = loss(seg_pred, seg_label) 
            #     total_loss += seg_loss.item()    
            # total_loss /= len(iter(testloader))
            # print('---------------------')
            # print('Validation seg loss: {}'.format(total_loss)) 

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
         
        # print('Lets eval...') 
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

        

        
        
        
        
         
        
        
        
        
        
            
         

        
        
        
        

        
        
        
            
            
            
            




        


        

    


            

    
    
    

    
    
    
    
    
        

    
    
    
    
    