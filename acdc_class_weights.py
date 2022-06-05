import torch  
import torch.multiprocessing as mp
from torch.utils import data 
import torchvision.transforms as transforms  
import os, sys 
import torch.backends.cudnn as cudnn 
from easydict import EasyDict as edict 
import numpy as np  
import time, datetime  
import random
from tqdm import tqdm 
from utils.optimize import * 
from PIL import Image  
import network 
import argparse 
from collections import defaultdict 


parser = argparse.ArgumentParser() 

parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID") 
parser.add_argument("--num_classes", type=int, default=19,
                        help="num classes (default: 19)")  
parser.add_argument("--data_dir", type=str, default='/home/sidd_s/scratch/dataset',
                        help='train data directory') 
parser.add_argument("--data_list", type=str, default='./dataset/list/acdc/acdc_valrgb.txt',
                        choices = ['./dataset/list/acdc/acdc_valrgb.txt', './dataset/list/acdc/acdc_trainrgb.txt'],help='list of names of validation files')  
parser.add_argument("--worker", type=int, default=4)   
parser.add_argument("--set", type=str, default='val')       

args = parser.parse_args()

def init_data(args): 
    loader = data.DataLoader(
            BaseDataSet(args, args.data_dir, args.data_list, args.num_classes, ignore_label=255, set=args.set),
            batch_size=1, shuffle=True, num_workers=args.worker, pin_memory=True)  
    return loader
    
class BaseDataSet(data.Dataset): 
    def __init__(self, args, root, list_path, num_class, ignore_label=255, set='val'): 

        self.root = root 
        self.list_path = list_path
        self.set = set
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
            
    def __len__(self):
        return len(self.files) 
    
    def __getitem__(self, index): 
        datafiles = self.files[index]
        name = datafiles["name"]    
        
        try:  
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
            print('**************') 
            print(index)
            index = index - 1 if index > 0 else index + 1 
            return self.__getitem__(index) 

        return image, label, name
                

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
    
    def validate(self): 
        loader = init_data(self.args)    
        label_count_ratio = defaultdict(int)
        print('Initialising the label count ratio:>>')
        
        for _, batch in tqdm(enumerate(loader)):
            image, seg_label, name = batch
            name = name[0].split('/')[-1]  
            seg_label = seg_label.squeeze() 
            
            for cls in range(19):
                label_sum = torch.sum(seg_label==cls).item() 
                label_count_ratio[cls] += label_sum 
        
        for cls in range(19):
            label_count_ratio[cls] /= (len(loader) * seg_label.shape[0]*seg_label.shape[1])

        print(label_count_ratio)

    
                         
        

class predictor(BaseTrainer):
    def __init__(self, args):
        self.args = args
       
    def predict(self):        
        self.validate() 
    
    
def main(args): 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id 
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    cudnn.enabled = True
    cudnn.benchmark = True   
    pred = predictor(args=args) 
    pred.predict()
    
    
    

if __name__ == "__main__":  
    mp.set_start_method('spawn')   ## for different random value using np.random  
    main(args)
