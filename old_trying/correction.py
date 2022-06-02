# from tkinter import image_names
# from unittest import TestLoader
import torch  
import torch.multiprocessing as mp

from torch.utils import data 
import torchvision.transforms as transforms  
import os, sys 
import torch.backends.cudnn as cudnn 
from model.unet_resnetenc import *
from model.unet_model import *
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
from tensorboardX import SummaryWriter 
from PIL import Image   
import network_city
from loader import cityscapes_loader, acdc_night_loader
from dataset.transforms import encode_labels

os.environ['CUDA_VISIBLE_DEVICES'] = '2' 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## ==============================================prior net define===================================================== 
def init_model(cfg):
    model = UNetWithResnet50Encoder(cfg.num_ip_channels, cfg.num_class) 
    # model = UNet(num_class=cfg.num_class)
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
    return model 


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
    img = np.array(img)
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

## =================================================Model Predictions as input======================================== 

class Model_pred(): 
    def __init__(self, config):
        self.cfg = config  
    
    def cityscapes_pred(self, images): 
        
        with torch.no_grad():
            model = network_city.modeling.__dict__['deeplabv3plus_resnet101'](num_classes=19, output_stride=16) 
            checkpoint = torch.load(self.cfg.ckpt, map_location=device)
            model.load_state_dict(checkpoint["model_state"]) 
            # model = nn.DataParallel(model)   ## to use only when we want to use multiple gpus
            model.to(device) 
            model.eval()   ## always in eval state cause we need to perform inference from the already pretrained model
            return model(images.float().cuda()) 
    
## ===================================================Training and validation================================================

class Trainer():
    def __init__(self, model, config, writer):
        self.model = model
        self.config = config
        self.writer = writer
        self.da_model, self.lightnet, self.weights = pred(self.config.num_class, self.config.model_dannet, self.config.restore_from_da, self.config.restore_light_path)  
        self.da_model = self.da_model.eval() 
        self.lightnet = self.lightnet.eval() 
        self.interp = nn.Upsample(size=(768, 768), mode='bilinear', align_corners=True) ## changing
        self.interp_whole = nn.Upsample(size=(1080, 1920), mode='bilinear', align_corners=True)
        
    def iter(self, batch):
        images, seg_labels, names = batch
        ind_cs = [] 
        ind_ac = []  
        for ind, nm in enumerate(names):
            if 'leftImg8bit' in nm:
                ind_cs.append(ind) 
            else:
                ind_ac.append(ind)

        # print(len(ind_cs),  len(ind_ac))
        if len(ind_cs)!=0:
            images_cs =  images[ind_cs] 
            seg_labels_cs = seg_labels[ind_cs]  
            city_model = Model_pred(self.config)
            seg_tensor = city_model.cityscapes_pred(images_cs)  
            label_perturb_tensor = seg_tensor.detach().clone() 
            seg_labels_cs = torch.tensor(encode_labels(seg_labels_cs))    

            ## one hot adding up
            if self.config.one_hot_ip: 
                pass

            seg_pred = self.model(label_perturb_tensor.float().cuda()) ## prediction by prior net 
            seg_labels_cs = seg_labels_cs.long().cuda() # cross entropy     
            loss = nn.CrossEntropyLoss(ignore_index= 255) 
            # print(seg_label.shape, seg_pred.shape) # torch.Size([1, 768, 768]) torch.Size([1, 19, 512, 512])
            seg_loss_cs = loss(seg_pred, seg_labels_cs)   
            
            
        if len(ind_ac)!=0: 
            images_ac =  images[ind_ac] 
            seg_labels_ac = seg_labels[ind_ac] 

            ## perturbation 
            with torch.no_grad():
                r = self.lightnet(images_ac.cuda())  
                enhancement = images_ac.cuda() + r
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
            label_perturb_tensor = seg_tensor.detach().clone() 
            label_perturb_tensor  = self.interp(label_perturb_tensor) 

            seg_pred = self.model(label_perturb_tensor.float().cuda()) ## prediction by prior net 
            seg_labels_ac = seg_labels_ac.long().cuda() # cross entropy     
            loss = nn.CrossEntropyLoss(ignore_index= 255) 
            # print(seg_label.shape, seg_pred.shape) # torch.Size([1, 768, 768]) torch.Size([1, 19, 512, 512])
            seg_loss_ac = loss(seg_pred, seg_labels_ac)   


        if self.config.rgb_save: 
            seg_tensor = torch.argmax(seg_tensor, dim=1) 
            seg_tensor = [torch.tensor(label_img_to_color(seg_tensor[sam])) for sam in range(seg_tensor.shape[0])]  
            seg_tensor = torch.stack(seg_tensor, dim=0)  
            seg_tensor = seg_tensor.permute(0,3,1,2)   
            # print(seg_tensor.shape) # torch.Size([8, 3, 512, 512])

        # seg_pred = self.model(label_perturb_tensor.float().cuda()) ## prediction by prior net 
        # seg_labels = seg_labels.long().cuda() # cross entropy     
        # loss = nn.CrossEntropyLoss(ignore_index= 255) 
        # # print(seg_label.shape, seg_pred.shape) # torch.Size([1, 768, 768]) torch.Size([1, 19, 512, 512])
        # seg_loss = loss(seg_pred, seg_labels)   
        
        if len(ind_ac)!=0 and len(ind_cs)!=0: 
            seg_loss = seg_loss_cs + seg_loss_ac  
        elif len(ind_ac)== 0 and len(ind_cs)!=0: 
            seg_loss = seg_loss_cs
        elif len(ind_ac)!=0 and len(ind_cs)==0: 
            seg_loss = seg_loss_ac 
        
        # print('>>>>>>>>>>>>') 
        # print(ind_cs, ind_ac)

        ## 

        self.losses.seg_loss = seg_loss  
        loss = seg_loss  
        loss.backward()    

    def train(self):
        writer = SummaryWriter(comment=self.config.model)
        
        if self.config.multigpu:       
            self.optim = optim.SGD(self.model.module.optim_parameters(self.config.learning_rate),
                          lr=self.config.learning_rate, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        else:
            # self.optim = optim.SGD(self.model.parameters(),
            #               lr=self.config.learning_rate, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
            self.optim = optim.Adam(self.model.parameters(),
                        lr=self.config.learning_rate)
    
        self.train_city = cityscapes_loader.city_train_data(self.config)  
        self.train_acdc_night = acdc_night_loader.acdc_night_train_data(self.config)  
        concat_dataset = data.ConcatDataset([self.train_city, self.train_acdc_night]) 
        self.train_loader = data.DataLoader(concat_dataset, batch_size=self.config.batch_size, shuffle=self.config.shuffle, num_workers=self.config.worker, pin_memory=True)
        # print(len(self.train_loader)) 
        # import pdb; pdb.set_trace()

        cu_iter = 0 
        early_stop_patience = 0 
        best_val_epoch_loss = float('inf') 
        train_epoch_loss = 0 
        max_iter = self.config.epochs * self.config.batch_size 
        print('Lets ride...') 

        for epoch in range(self.config.epochs):
            epoch_infor = ('epoch = {:6d}/{:6d}, exp = {}'.format(epoch, int(self.config.epochs), self.config.note))
            
            print(epoch_infor) 

            for i_iter, batch in tqdm(enumerate(self.train_loader)): 

                # print(name)  
                # cu_iter +=1
                # adjust_learning_rate(self.optim, cu_iter, max_iter, self.config)  
                self.optim.zero_grad()
                self.losses = edict({}) 
                losses = self.iter(batch) 

                train_epoch_loss += self.losses['seg_loss'].item()
                print('train_iter_loss:', self.losses['seg_loss'].item())
                self.optim.step() 

            train_epoch_loss /= len(iter(self.train_loader))  

            if self.config.val:

                loss_infor = 'train loss :{:.4f}'.format(train_epoch_loss)
                print(loss_infor)

                valid_epoch_loss = self.validate(epoch)

                writer.add_scalars('Epoch_loss',{'train_tensor_epoch_loss': train_epoch_loss,'valid_tensor_epoch_loss':valid_epoch_loss},epoch)  

                if(valid_epoch_loss < best_val_epoch_loss):
                    early_stop_patience = 0 ## early stopping variable 
                    best_val_epoch_loss = valid_epoch_loss
                    print('********************')
                    print('best_val_epoch_loss: ', best_val_epoch_loss)
                    print("MODEL UPDATED")
                    name = self.config['model'] + '.pth' # for the ce loss 
                    torch.save(self.model.state_dict(), osp.join(self.config["snapshot"], name))
                # else: ## early stopping with patience of 50
                #     early_stop_patience = early_stop_patience + 1 
                #     if early_stop_patience == 50:
                #         break
                self.model = self.model.train()
            
            # if early_stop_patience == 50: 
            #     print('Early_Stopping!!!')
            #     break 


    def validate(self, epoch): 
        self.model = self.model.eval().cuda()
        total_loss = 0
        # testloader = init_val_data(self.config)   
        val_city = cityscapes_loader.city_val_data(self.config)     
        val_acdc = acdc_night_loader.acdc_night_val_data(self.config) 
        val_concat_dataset = data.ConcatDataset([val_city, val_acdc]) 
        self.val_loader = data.DataLoader(val_concat_dataset, batch_size=1, shuffle=self.config.shuffle, num_workers=self.config.worker, pin_memory=True)
        
        # print(len(iter(self.val_loader)))

        print('In validation')   
        for i_iter, batch in tqdm(enumerate(self.val_loader)):
            image, seg_label, name = batch 
            
            name = name[0] ## batch size 1 so only one name in the batch 

            if 'leftImg8bit' in name:    
                city_model = Model_pred(self.config) 
                seg_tensor = city_model.cityscapes_pred(image)  
                label_perturb_tensor = seg_tensor.detach().clone() 
                seg_pred = self.model(label_perturb_tensor.float().cuda())  
                seg_label = torch.tensor(encode_labels(seg_label))  

            else:
                ## perturbation 
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

                output2 = self.interp_whole(output2).cpu().numpy() 
                seg_tensor = torch.tensor(output2) 

                # print(pred.shape)  
                seg_tensor = F.softmax(seg_tensor, dim=1)   

                if self.config.rgb_save:             
                    seg_tensor = torch.argmax(seg_tensor, dim=1) 
                    seg_tensor = [torch.tensor(label_img_to_color(seg_tensor[sam], 'pred_dannet_bdd_city/' + str(sam + iter) + '.png')) for sam in range(seg_tensor.shape[0])]  
                    # seg_tensor = torch.stack(seg_tensor, dim=0)  
                    # seg_tensor = seg_tensor.permute(0,3,1,2)   

                label_perturb_tensor = seg_tensor.detach().clone() 
                seg_pred = self.model(label_perturb_tensor.float().cuda())    
                seg_pred = self.interp_whole(seg_pred)
            
            
            seg_label = seg_label.long().cuda() # cross entropy  
            # print(torch.unique(seg_label).shape) ## > 19 unique values that is why failing ;) # now OK  
            loss = nn.CrossEntropyLoss(ignore_index= 255)    
            seg_loss = loss(seg_pred, seg_label)      
            total_loss += seg_loss.item() 
                

        total_loss /= len(iter(self.val_loader))
        print('---------------------')
        print('Validation seg loss: {} at epoch {}'.format(total_loss,epoch))
        return total_loss





def main(): 
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    cudnn.enabled = True
    cudnn.benchmark = True   
    config, writer = init_config("config/config_exp.yml", sys.argv)
    model = init_model(config)
    trainer = Trainer(model, config, writer)
    trainer.train() 



if __name__ == "__main__": 
    mp.set_start_method('spawn')   ## for different random value using np.random 
    start = datetime.datetime(2020, 1, 22, 23, 00, 0)
    print("wait")
    while datetime.datetime.now() < start:
        time.sleep(1)
    main()
