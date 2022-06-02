import torch  
import torch.multiprocessing as mp

from torch.utils import data 
import torchvision.transforms as transforms  
import os, sys 
import torch.backends.cudnn as cudnn 
from easydict import EasyDict as edict 
import numpy as np  
import random  
import time, datetime  
import torch.nn as nn 
from tqdm import tqdm 
import torch.nn.functional as F
from utils.optimize import * 
import  torch.optim as optim 
from tensorboardX import SummaryWriter 
from PIL import Image  
import network 
import argparse 
from old_trying.model.unet_resnet18 import * 


parser = argparse.ArgumentParser() 

parser.add_argument("--gpu_id", type=str, default='1',
                        help="GPU ID") 
parser.add_argument("--num_classes", type=int, default=3,
                        help="num classes (default: 3) taking as colorisation work")  
parser.add_argument("--num_ip_channels", '-ip_chs',type=int, default=3,
                        help="num of input channels (default: 3)")  
parser.add_argument("--restore_from", type=str, default='None',
                        help='prior net trained model path') 
parser.add_argument("--multigpu", '-mul',action='store_true', default=False,
                        help="multigpu training") 
parser.add_argument("--num_patch_perturb", type=int, default=100,  # may be req 
                        help="num of patches to be perturbed") 
parser.add_argument("--patch_size", type=int, default=10, # may be req 
                        help="size of square patch to be perturbed")
parser.add_argument("--save_writer_name", '-svnm', type=str, default='deeplabv3+_synthetic_cityscapes_acdc_manual_color_mapping',
                        help='saving tensorboard file name')  # may be req  
parser.add_argument("--learning_rate", '-lr', type=float, default=1e-2, # may be req  
                        help="learning rate for trainig prior net") 
parser.add_argument("--momentum", '-m', type=float, default=0.9, 
                        help="momentum for sgd optimiser") 
parser.add_argument("--weight_decay", '-wd', type=float, default=0.0005,
                    help="weight_decay for sgd optimiser") 
parser.add_argument("--train_data_dir", '-tr_dr', type=str, default='/home/sidd_s/scratch/dataset',
                        help='train data directory') 
parser.add_argument("--train_data_list", '-tr_dr_lst', type=str, default='./dataset/list/acdc/acdc_trainrgb.txt',
                        choices = ['./dataset/list/acdc/acdc_trainrgb.txt'], help='list of names of training files')  
parser.add_argument("--val_data_dir", '-val_dr', type=str, default='/home/sidd_s/scratch/dataset',
                        help='train data directory') 
parser.add_argument("--val_data_list", '-val_dr_lst', type=str, default='./dataset/list/acdc/acdc_valrgb.txt',
                        choices = ['./dataset/list/acdc/acdc_valrgb.txt'],help='list of names of validation files')  
parser.add_argument("--input_size", '-ip_sz', type=int, nargs="+", default=[1080, 1920],
                        help='actual input size')   
parser.add_argument("--batch_size", type=int, default=4)  # may be req 
parser.add_argument("--worker", type=int, default=4)   
parser.add_argument("--train_dataset", '-tr_nm', type=str, default='synthetic_acdc_train_label',
                        choices = ['acdc_train_label','synthetic_manual_acdc_train_label', 'synthetic_acdc_train_label'], help='train datset name') 
parser.add_argument("--val_dataset", '-val_nm', type=str, default='synthetic_manual_acdc_val_label',
                        choices = ['acdc_val_label', 'synthetic_manual_acdc_val_label', 'synthetic_acdc_val_label'], help='valditation datset name') 
parser.add_argument("--print_freq", type=int, default=1)  
parser.add_argument("--epochs", type=int, default=500)   # may be req 
parser.add_argument("--snapshot", type=str, default='../scratch/saved_models/acdc/dannet',
                        help='model saving directory')   
parser.add_argument("--exp", '-e', type=int, default=1, help='which exp to run')                     
parser.add_argument("--synthetic_perturb", type=str, default='synthetic_cityscapes_acdc', # using synthetic cityscapes manual color of acdc night images w/o any perturbation for learning first color mapping only then performing fine tunnnig over it when noise are introduced in it.
                        help='evalutating technique')
parser.add_argument("--end_learning_rate", type=float, default=1e-3, # may be req  
                        help="end learning rate required in the lr scheduling") 
parser.add_argument("--power", type=float, default=0.9,  
                        help="power to be used in learning rate scheduling")    

args = parser.parse_args()


## dataset init 
def init_train_data(args): 
    
    trainloader = data.DataLoader(
            BaseDataSet(args, args.train_data_dir, args.train_data_list, args.train_dataset, args.num_classes, ignore_label=255, set='train'),
            batch_size=args.batch_size, shuffle=True, num_workers=args.worker, pin_memory=True)

    return trainloader 

def init_val_data(args): 

    valloader = data.DataLoader(
            BaseDataSet(args, args.val_data_dir, args.val_data_list, args.val_dataset, args.num_classes, ignore_label=255, set='val'),
            batch_size=1, shuffle=True, num_workers=args.worker, pin_memory=True)

    return valloader
    
class BaseDataSet(data.Dataset): 
    def __init__(self, args, root, list_path,dataset, num_class, ignore_label=255, set='val'): 

        self.root = root 
        self.list_path = list_path
        self.set = set
        self.dataset = dataset 
        self.num_class = num_class 
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
            img_file = os.path.join(self.root, name.split('night')[0], 'night/synthetic', self.set, self.args.synthetic_perturb, name.split('/')[-1])
            # replace = (("_rgb_anon", "_gt_labelTrainIds"), ("acdc_trainval", "acdc_gt"), ("rgb_anon", "gt"))
            replace = (("_rgb_anon", "_gt_labelColor"), ("acdc_trainval", "acdc_gt"), ("rgb_anon", "gt"))
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
        
        transforms_compose_train = transforms.Compose([
                        transforms.Resize((512, 512)),
                        transforms.ToTensor() 
                    ]) 
        transforms_compose_train_lb = transforms.Compose([ 
                        transforms.Resize((512, 512)) 
                    ])
        
        transforms_compose_val = transforms.Compose([
                        transforms.ToTensor(), 
                    ])
        
        # transforms_compose_val_lb = transforms.Compose([ 
        #                 transforms.ToTensor() 
        #             ]) 
        
        try:       
            if self.set == 'val':
                image = Image.open(datafiles["img"]).convert('RGB')   
                image = transforms_compose_val(image)    
                image = torch.tensor(np.array(image)).float() 
                  
                label = Image.open(datafiles["label"])  
                # label = transforms_compose_val(label)
                label = torch.tensor(np.array(label))  
            else:
                image = Image.open(datafiles["img"]).convert('RGB')   
                image = transforms_compose_train(image)    
                image = torch.tensor(np.array(image)).float()    

                label = Image.open(datafiles["label"]) 
                label = transforms_compose_train_lb(label) 
                label = torch.tensor(np.array(label))  
                    
        except: 
            print('**************') 
            print(index)
            index = index - 1 if index > 0 else index + 1 
            return self.__getitem__(index) 

        return image, label, name
                
def init_model(args): 
    ## auto encoder style for correction 
    
    model = ResNet18UNet(n_ip_channels=args.num_ip_channels, n_class=args.num_classes) # n_class is the number of output channels  
        
    if args.restore_from != 'None':
        params = torch.load(args.restore_from)['model_state']
        model.load_state_dict(params)
        print('----------Model initialize with weights from-------------: {}'.format(args.restore_from))
    if args.multigpu:
        model = nn.DataParallel(model)
        
    model.train().cuda()
    print('Mode --> Train') 
    return model

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
        20: [255, 255, 255]
        } 
    img = np.array(img.cpu())
    img_height, img_width = img.shape
    img_color = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    for row in range(img_height):
        for col in range(img_width):
            label = img[row][col] 
            img_color[row, col] = np.array(label_to_color[label])  
    return img_color


class BaseTrainer(object):
    def __init__(self, models, optimizers, loaders, args,  writer):
        self.model = models
        self.optim = optimizers
        self.loader = loaders
        self.args = args
        self.output = {}
        self.writer = writer

    def forward(self):
        pass
    def backward(self):
        pass

    def iter(self):
        pass
    
    def validate(self, epoch): 
        self.model = self.model.eval() 
        total_loss = 0
        testloader = init_val_data(self.args)
        for i_iter, batch in tqdm(enumerate(testloader)):
            image, seg_label, name = batch 
            seg_pred = self.model(image.float().cuda()) 
            seg_label = seg_label.float().cuda()   
            loss = nn.MSELoss()  
            seg_loss = loss(seg_pred, seg_label)   
            total_loss += seg_loss.item()

        total_loss /= len(iter(testloader))
        print('---------------------')
        print('Validation seg loss: {} at epoch {}'.format(total_loss,epoch))
        return total_loss


class Trainer(BaseTrainer):
    def __init__(self, model, args):
        self.model = model
        self.args = args
        
    def iter(self, batch):
        
        image , seg_label, name = batch  
        seg_pred = self.model(image.float().cuda()).squeeze() 
        seg_label = seg_label.float().cuda()    
        loss = nn.MSELoss() 
        seg_loss = loss(seg_pred, seg_label)   
        self.losses.seg_loss = seg_loss 
        loss = seg_loss  
        loss.backward()   

    def train(self):
        writer = SummaryWriter(comment=self.args.save_writer_name)
        
        if self.args.multigpu:       
            self.optim = optim.SGD(self.model.parameters(),
                          lr=self.args.learning_rate, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
            # self.optim = optim.Adam(self.model.parameters(),
            #             lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        else:
            self.optim = optim.SGD(self.model.parameters(),
                          lr=self.args.learning_rate, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
            # self.optim = optim.Adam(self.model.parameters(),
            #             lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        
        self.loader = init_train_data(self.args)  
        
        best_val_epoch_loss = float('inf') 
        train_epoch_loss = 0 
        max_decay_steps = self.args.epochs
        print('Lets ride...') 

        for epoch in tqdm(range(self.args.epochs)):
            epoch_infor = ('epoch = {:6d}/{:6d}, exp = {}'.format(epoch, int(self.args.epochs), self.args.save_writer_name))
            
            print(epoch_infor)
            scheduler = PolynomialLRDecay(self.optim, max_decay_steps=max_decay_steps, args=self.args) 
            scheduler.step(epoch) 

            for i_iter, batch in tqdm(enumerate(self.loader)):
                self.optim.zero_grad()
                self.losses = edict({})
                self.iter(batch) 

                train_epoch_loss += self.losses['seg_loss'].item()
                print('train_iter_loss:', self.losses['seg_loss'].item())
                self.optim.step()  

            train_epoch_loss /= len(iter(self.loader))  

            loss_infor = 'train loss :{:.4f}'.format(train_epoch_loss)
            print(loss_infor) 

            valid_epoch_loss = self.validate(epoch)

            writer.add_scalars('Epoch_loss',{'train_tensor_epoch_loss': train_epoch_loss,'valid_tensor_epoch_loss':valid_epoch_loss},epoch)  

            if(valid_epoch_loss < best_val_epoch_loss):
                best_val_epoch_loss = valid_epoch_loss
                best_epoch = epoch
                print('********************') 
                print("MODEL UPDATED")
                name = self.args.save_writer_name + '.pth' # for the ce loss  
                torch.save(self.model.state_dict(), os.path.join(self.args.snapshot, name))
                print(os.path.join(self.args.snapshot, name))
                            
            self.model = self.model.train()
        
        print('best_val_epoch_loss: ', best_val_epoch_loss, 'best_epoch:', best_epoch)
        writer.export_scalars_to_json("./all_scalars.json")
        writer.close()


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
    trainer.train() 


if __name__ == "__main__":  
    mp.set_start_method('spawn')   ## for different random value using np.random  
    main(args)

        

        
        
        
        
         
        
        
        
        
        
            
            
        
        
            
         

        
        
        
        

        
        
        
            
            
            
            




        


        

    


            

    
    
    

    
    
    
    
    
        

    
    
    
    
    