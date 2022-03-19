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
from tensorboardX import SummaryWriter 
from PIL import Image  
from dataset.network.dannet_pred import pred 


## ==============================================prior net define===================================================== 
def init_model(cfg):
    # model = UNet_mod(cfg.num_channels, cfg.num_class, cfg.small).cuda() ## previous unet model   
    # unet = cfg.unet
    # model = UNet(unet.enc_chs, unet.dec_chs, unet.num_class).cuda()  
    model = UNetWithResnet50Encoder(cfg.num_ip_channels, cfg.num_class)
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
    def __init__(self, config, cityscapes_loader, acdc_night_loader):
            

        pass  

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

        if self.dataset == 'acdc_city':  

            ## acdc read txt 
            with open(self.list_path[0]) as f: 
                for item in f.readlines(): 
                    fields = item.strip().split('\t')[0]
                    self.img_ids.append(fields)  
            
            ## city read txt 
            with open(self.list_path[2]) as f: 
                for item in f.readlines(): 
                    fields = item.strip().split('\t')[0]
                    self.img_ids.append(fields)

            for name in self.img_ids:      
                ## city 
                if len(name.split('/')) == 2:
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


    def __len__(self):
        return len(self.files) 
    
    def __getitem__(self, index): 
        datafiles = self.files[index]
        name = datafiles["name"]    
        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  ## image net mean and std 
    
        try:  
            if self.dataset in ['acdc_city']:  
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



## ==================================================Training and validation================================================

class Trainer():
    def __init__(self, model, config, writer):
        self.model = model
        self.config = config
        self.writer = writer
        self.da_model, self.lightnet, self.weights = pred(self.config.num_class, self.config.model_dannet, self.config.restore_from_da, self.config.restore_light_path)  
        self.da_model = self.da_model.eval() 
        self.lightnet = self.lightnet.eval() 
        self.interp = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)
        self.interp_whole = nn.Upsample(size=(1080, 1920), mode='bilinear', align_corners=True)
        
    def iter(self, batch):
        image , seg_label, name = batch 

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

        output2 = self.interp(output2).cpu().numpy() 
        seg_tensor = torch.tensor(output2)  

        # print(seg_tensor.shape) # torch.Size([8, 19, 512, 512])   
        seg_tensor = F.softmax(seg_tensor, dim=1)   

        if self.config.rgb: 
            
            seg_tensor = torch.argmax(seg_tensor, dim=1) 
            seg_tensor = [torch.tensor(label_img_to_color(seg_tensor[sam])) for sam in range(seg_tensor.shape[0])]  
            seg_tensor = torch.stack(seg_tensor, dim=0)  
            seg_tensor = seg_tensor.permute(0,3,1,2)   
            # print(seg_tensor.shape) # torch.Size([8, 3, 512, 512])
      
        label_perturb_tensor = seg_tensor.detach().clone()
        seg_pred = self.model(label_perturb_tensor.float().cuda()) 

        seg_pred  = self.interp(seg_pred)  
        
        seg_label = seg_label.long().cuda() # cross entropy    
        loss = nn.CrossEntropyLoss(ignore_index= 255)

        seg_loss = loss(seg_pred, seg_label)   
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
        
        valid_epoch_loss = self.validate(0) 

        self.loader, _ = init_train_data(self.config) 

        cu_iter = 0 
        early_stop_patience = 0 
        best_val_epoch_loss = float('inf') 
        train_epoch_loss = 0 
        max_iter = self.config.epochs * self.config.batch_size 
        print('Lets ride...')

        for epoch in range(self.config.epochs):
            epoch_infor = ('epoch = {:6d}/{:6d}, exp = {}'.format(epoch, int(self.config.epochs), self.config.note))
            
            print(epoch_infor)
 
            for i_iter, batch in tqdm(enumerate(self.loader)):
                cu_iter +=1
                # adjust_learning_rate(self.optim, cu_iter, max_iter, self.config)
                self.optim.zero_grad()
                self.losses = edict({})
                losses = self.iter(batch) 

                train_epoch_loss += self.losses['seg_loss'].item()
                print('train_iter_loss:', self.losses['seg_loss'].item())
                self.optim.step() 

            train_epoch_loss /= len(iter(self.loader))  

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
        self.model = self.model.eval() 
        total_loss = 0
        testloader = init_val_data(self.config)  
        print('In validation') 
        iter = 0
        # print(len(iter(testloader)))
        for i_iter, batch in tqdm(enumerate(testloader)):
            image, seg_label, name = batch 

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

            if self.config.rgb:             
                iter = iter + 1 
                seg_tensor = torch.argmax(seg_tensor, dim=1) 
                seg_tensor = [torch.tensor(label_img_to_color(seg_tensor[sam], 'pred_dannet_bdd_city/' + str(sam + iter) + '.png')) for sam in range(seg_tensor.shape[0])]  
                # seg_tensor = torch.stack(seg_tensor, dim=0)  
                # seg_tensor = seg_tensor.permute(0,3,1,2)   

        #     label_perturb_tensor = seg_tensor.detach().clone() 
        #     seg_pred = self.model(label_perturb_tensor.float().cuda())    
        #     seg_pred = self.interp_whole(seg_pred)

        #     seg_label = seg_label.long().cuda() # cross entropy   
        #     loss = nn.CrossEntropyLoss(ignore_index= 255) 
        #     seg_loss = loss(seg_pred, seg_label) 
        #     total_loss += seg_loss.item() 

        # total_loss /= len(iter(testloader))
        # print('---------------------')
        # print('Validation seg loss: {} at epoch {}'.format(total_loss,epoch))
        # return total_loss
        return




def main(): 
    os.environ['CUDA_VISIBLE_DEVICES'] = '2' 
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
