import torch 
from torch.utils import data 
import os.path as osp
from PIL import Image
import numpy as np 
import torchvision.transforms as transforms

class acdcnightloader(data.Dataset): 

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
            img_file = osp.join(self.root, name) 
            replace = (("_rgb_anon", "_gt_labelTrainIds"), ("acdc_trainval", "acdc_gt"), ("rgb_anon", "gt"))
            nm = name
            for r in replace: 
                nm = nm.replace(*r) 
            label_file = osp.join(self.root, nm)   
            # print('****')
            # print(name)
            self.files.append({
                                "img": img_file,
                                "label":label_file,
                                "name": name
                            }) 
        
        # print('*******') 
        # print(len(self.files))
        
    def __len__(self):
        return len(self.files) 

    def __getitem__(self, index: int): 
        datafiles = self.files[index]
        name = datafiles["name"]    
        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  ## image net mean and std 
        try: 
            if self.set=='val':   
                ## image for transforming  
                transforms_compose_img = transforms.Compose([
                    transforms.Resize((540, 960)),
                    transforms.ToTensor(),
                    transforms.Normalize(*mean_std), ## use only in training while doing validation  
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

                transforms_compose_img = transforms.Compose([transforms.Resize((768,768)), transforms.ToTensor(), transforms.Normalize(*mean_std)])  
                ## changing the traiing images and labels sizes so as to make it same as that of cityscapes 
                img_trans = transforms_compose_img(image) 
                image = torch.tensor(np.array(img_trans)).float() 
                
                transforms_compose_label = transforms.Compose([transforms.Resize((768,768),interpolation=Image.NEAREST)]) 
                label = Image.open(datafiles["label"]) 
                label = transforms_compose_label(label)   
                label = torch.tensor(np.array(label)) 
                # label = torch.LongTensor(np.array(label).astype('int32'))

        except: 
            # print('**************') 
            # print(index)
            index = index - 1 if index > 0 else index + 1 
            return self.__getitem__(index) 
        
        return image, label, name


def acdc_night_train_data(cfg): 
    train_env = cfg['acdc_train_label']  
    cfg.train_data_dir = train_env.data_dir
    cfg.train_data_list = train_env.data_list

    # trainloader = data.DataLoader(
    #         acdcnightloader(cfg, cfg.train_data_dir, cfg.train_data_list, cfg.train, cfg.num_class, ignore_label=255, set='train'),
    #         batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.worker, pin_memory=True)
    # return trainloader,cfg 
    
    acdc_train_data = acdcnightloader(cfg, cfg.train_data_dir, cfg.train_data_list, cfg.num_class, set='train') 
    # print('<<<<<<<<<<<<') 
    # print(len(acdc_train_data))
    return acdc_train_data


def acdc_night_val_data(cfg): 
    val_env = cfg['acdc_val_label'] 
    cfg.val_data_dir = val_env.data_dir
    cfg.val_data_list = val_env.data_list

    # valloader = data.DataLoader(
    #         acdcnightloader(cfg, cfg.val_data_dir, cfg.val_data_list, cfg.val, cfg.num_class, ignore_label=255, set='val'),
    #         batch_size=1, shuffle=True, num_workers=cfg.worker, pin_memory=True)
    # return valloader

    acdc_val_data = acdcnightloader(cfg, cfg.val_data_dir, cfg.val_data_list, cfg.num_class, set='val')  
    return acdc_val_data


        
        
        