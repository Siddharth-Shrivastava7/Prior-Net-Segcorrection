import torch 
from torch.utils import data 
import os.path as osp

class cityscapesloader(data.Dataset): 

    def __init__(self, cfg, root, list_path, num_class, ignore_label=255, set='val'):
        super().__init__()   
        
        self.root = root 
        self.list_path = list_path
        self.set = set
        self.num_class = num_class 
        self.ignore_label = ignore_label
        self.cfg = cfg  
        self.img_ids = []   
        self.files = []  

        with open(self.list_path) as f: 
            for item in f.readlines(): 
                fields = item.strip().split('\t')[0]
                self.img_ids.append(fields)

        for name in self.img_ids:      
            img_file = osp.join(self.root[2] ,'leftImg8bit/train', name)
            label_name = name.replace('leftImg8bit', 'gtFine_labelIds') 
            label_file = osp.join(self.root[2], 'gtFine/train',  label_name) 
        
        self.files.append({
                            "img": img_file,
                            "label":label_file,
                            "name": name
                        }) 
        
    def __len__(self):
        return len(self.files) 

    def __getitem__(self, index: int): 
        datafiles = self.files[index]
        name = datafiles["name"]    
        # mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  ## image net mean and std 
        
        



def city_train_data(cfg): 
    train_env = cfg[cfg.train]  
    cfg.train_data_dir = train_env.data_dir
    cfg.train_data_list = train_env.data_list

    trainloader = data.DataLoader(
            cityscapesloader(cfg, cfg.train_data_dir, cfg.train_data_list, cfg.train, cfg.num_class, ignore_label=255, set='train'),
            batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.worker, pin_memory=True)

    return trainloader,cfg

def city_val_data(cfg): 
    val_env = cfg[cfg.val] 
    cfg.val_data_dir = val_env.data_dir
    cfg.val_data_list = val_env.data_list

    valloader = data.DataLoader(
            cityscapesloader(cfg, cfg.val_data_dir, cfg.val_data_list, cfg.val, cfg.num_class, ignore_label=255, set='val'),
            batch_size=1, shuffle=True, num_workers=cfg.worker, pin_memory=True)

    return valloader


        
        
        