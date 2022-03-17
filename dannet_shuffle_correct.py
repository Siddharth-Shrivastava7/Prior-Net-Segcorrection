import torch  
import torch.multiprocessing as mp

from torch.utils import data 
import torchvision.transforms as transforms  
import os, sys 
import torch.backends.cudnn as cudnn 
from model.Unet import * 
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


## dataset init 
def init_train_data(cfg): 
    train_env = cfg[cfg.train] 
    cfg.train_data_dir = train_env.data_dir
    cfg.train_data_list = train_env.data_list
    cfg.input_size = train_env.input_size  

    trainloader = data.DataLoader(
            BaseDataSet(cfg, cfg.train_data_dir, cfg.train_data_list, cfg.train, cfg.num_classes, ignore_label=255, set='train'),
            batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.worker, pin_memory=True)

    return trainloader,cfg

def init_val_data(cfg): 
    val_env = cfg[cfg.val] 
    cfg.val_data_dir = val_env.data_dir
    cfg.val_data_list = val_env.data_list
    cfg.input_size = val_env.input_size  

    valloader = data.DataLoader(
            BaseDataSet(cfg, cfg.val_data_dir, cfg.val_data_list, cfg.val, cfg.num_classes, ignore_label=255, set='val'),
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
                    # print(img_trans)
                    image = torch.tensor(np.array(img_trans)).float() 
                    
                    transforms_compose_label = transforms.Compose([transforms.Resize((512,512),interpolation=Image.NEAREST)]) 
                    label = Image.open(datafiles["label"]) 
                    label = transforms_compose_label(label)   
                    label = torch.tensor(np.array(label))   
                    
        except: 
            print('**************') 
            print(index)
            index = index - 1 if index > 0 else index + 1 
            return self.__getitem__(index) 

        return image, label, name
                

def init_model(cfg):
    model = UNet_mod(cfg.num_channels, cfg.num_classes, cfg.small).cuda()
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

class BaseTrainer(object):
    def __init__(self, models, optimizers, loaders, config,  writer):
        self.model = models
        self.optim = optimizers
        self.loader = loaders
        self.config = config
        self.output = {}
        self.writer = writer

    def forward(self):
        pass
    def backward(self):
        pass

    def iter(self):
        pass

    def train(self):
        for i_iter in range(self.config.num_steps):
            losses = self.iter(i_iter)
            if i_iter % self.config.print_freq ==0:
                self.print_loss(i_iter)
            if i_iter % self.config.save_freq ==0 and i_iter != 0:
                self.save_model(i_iter)
            if self.config.val and i_iter % self.config.val_freq ==0 and i_iter!=0:
                self.validate()

    def save_model(self, iter):
        tmp_name = '_'.join((self.config.train, str(iter))) + '.pth'
        torch.save(self.model.state_dict(), os.path.join(self.config['snapshot'], tmp_name))

    def print_loss(self, iter):
        iter_infor = ('iter = {:6d}/{:6d}, exp = {}'.format(iter, self.config.num_steps, self.config.note))
        to_print = ['{}:{:.4f}'.format(key, self.losses[key].item()) for key in self.losses.keys()]
        loss_infor = '  '.join(to_print)
        if self.config.screen:
            print(iter_infor +'  '+ loss_infor)
        if self.config.tensorboard and self.writer is not None:
            for key in self.losses.keys():
                self.writer.add_scalar('train/'+key, self.losses[key], iter)
    
    def validate(self, epoch): 
        self.model = self.model.eval() 
        total_loss = 0
        testloader = init_val_data(self.config)
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
            
            randx_pix = np.random.randint(seg_tensor.shape[2]-10)
            randy_pix = np.random.randint(seg_tensor.shape[2]-10)
            # print(randx_pix, randy_pix) 
            
            for x in range(10):
                for y in range(10):  
                    org_batch_val = seg_tensor[:, :, randx_pix + x, randy_pix + y]  
                    shuffle_inds = torch.randperm(org_batch_val.shape[1]) 
                    # print(shuffle_inds)   
                    seg_tensor[:, :, randx_pix + x, randy_pix + y] = org_batch_val[:, shuffle_inds] 
        
            label_perturb_tensor = seg_tensor.detach().clone() 
            seg_pred = self.model(label_perturb_tensor.float().cuda())    

            seg_label = seg_label.long().cuda() # cross entropy   
            loss = CrossEntropy2d() # ce loss  
            seg_loss = loss(seg_pred, seg_label) 
            total_loss += seg_loss.item()

        total_loss /= len(iter(testloader))
        print('---------------------')
        print('Validation seg loss: {} at epoch {}'.format(total_loss,epoch))
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
        self.da_model, self.lightnet, self.weights = pred(self.config.num_classes, self.config.model_dannet, self.config.restore_from_da, self.config.restore_light_path)  
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

        # print(pred.shape)  
        seg_tensor = F.softmax(seg_tensor, dim=1)  

        # seg_tensor_org = seg_tensor.clone()   
        
        randx_pix = np.random.randint(seg_tensor.shape[2]-10)
        randy_pix = np.random.randint(seg_tensor.shape[2]-10)
        # print(randx_pix, randy_pix) 
        
        for x in range(10):
            for y in range(10):  
                org_batch_val = seg_tensor[:, :, randx_pix + x, randy_pix + y]  
                shuffle_inds = torch.randperm(org_batch_val.shape[1]) 
                # print(shuffle_inds)   
                seg_tensor[:, :, randx_pix + x, randy_pix + y] = org_batch_val[:, shuffle_inds] 

        # max_org = torch.argmax(seg_tensor_org[0], dim=0) 
        # max_mod = torch.argmax(seg_tensor[0], dim=0)
        # print(torch.sum(max_org!=max_mod).item())  # 95 yes! 
 
        label_perturb_tensor = seg_tensor.detach().clone()
        seg_pred = self.model(label_perturb_tensor.float().cuda()) 
        # print(seg_label)  
        seg_label = seg_label.long().cuda() # cross entropy  
        loss = CrossEntropy2d() # ce loss  
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
            self.optim = optim.SGD(self.model.parameters(),
                          lr=self.config.learning_rate, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
            # self.optim = optim.Adam(self.model.parameters(),
            #             lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        
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

        writer.export_scalars_to_json("./all_scalars.json")
        writer.close()



def main(): 
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
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

        

        
        
        
        
         
        
        
        
        
        
            
            
        
        
            
         

        
        
        
        

        
        
        
            
            
            
            




        


        

    


            

    
    
    

    
    
    
    
    
        

    
    
    
    
    