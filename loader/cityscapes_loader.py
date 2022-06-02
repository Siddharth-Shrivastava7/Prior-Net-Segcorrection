import torch 
from torch.utils import data 
import os.path as osp
from PIL import Image
import numpy as np 
import torchvision.transforms as transforms 
# import ext_transforms_city as et
from loader import ext_transforms_city as et  ## from where we are running the code their import module 

## defining global train and val transforms 

train_transform = et.ExtCompose([
                    # et.ExtResize( 512 ),
                    et.ExtRandomCrop(size=(768, 768)),
                    et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                    et.ExtRandomHorizontalFlip(),
                    et.ExtToTensor(),
                    et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
                ]) 

val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])


## dictionary 


class cityscapesloader(data.Dataset): 

    def __init__(self, cfg, root, list_path, num_class, ignore_label=255, set='val', train_transform = None,
                val_transform = None):
        super().__init__()   

        # self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33] 
        # self.class_map = dict(zip(self.valid_classes, range(19))) ## have to continue from here  
        # print(self.class_map)    

        self._key = np.array([-1, -1, -1, -1, -1, -1,
                              -1, -1, 0, 1, -1, -1,
                              2, 3, 4, -1, -1, -1,
                              5, -1, 6, 7, 8, 9,
                              10, 11, 12, 13, 14, 15,
                              -1, -1, 16, 17, 18])
        self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32') 
         
        self.root = root 
        self.list_path = list_path
        self.set = set
        self.num_class = num_class 
        self.ignore_label = ignore_label
        self.cfg = cfg  
        self.img_ids = []   
        self.files = []   
        self.train_transform = train_transform 
        self.val_transform = val_transform

        with open(self.list_path) as f: 
            for item in f.readlines(): 
                fields = item.strip().split('\t')[0]
                self.img_ids.append(fields) 
        

        for name in self.img_ids:       
            img_file = osp.join(self.root ,'leftImg8bit',self.set, name)
            label_name = name.replace('leftImg8bit', 'gtFine_labelIds') 
            label_file = osp.join(self.root, 'gtFine', self.set, label_name)   
            self.files.append({
                                "img": img_file,
                                "label":label_file,
                                "name": name
                            }) 

    def __len__(self):
        return len(self.files)  

    def _class_to_index(self, mask):
        # assert the value
        values = np.unique(mask)  
        # print('>>>>>>>') 
        # print(values)
        for value in values:
            assert (value in self._mapping) 
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape) 
    
    def _mask_transform(self, mask): 
        # print('>>>>>>>')
        target = self._class_to_index(np.array(mask).astype('int32')) 
        return torch.LongTensor(np.array(target).astype('int32'))  
        # return torch.tensor(np.array(target))

    def __getitem__(self, index: int): 
        datafiles = self.files[index]
        name = datafiles["name"]     

        try: 
            if self.set=='val':   
                ## image for transforming   
                image = Image.open(datafiles["img"]).convert('RGB')                              
                label = Image.open(datafiles["label"])    
                image, label = self.val_transform(image, label)
                # label = self._mask_transform(label)    
                # print(torch.unique(label)) 

            else: 
                image = Image.open(datafiles["img"]).convert('RGB')  
                label = Image.open(datafiles["label"]) 
                image, label = self.train_transform(image, label)  
                # label = self._mask_transform(label)   
                # print(torch.unique(label)) 
            
            ## have to change label ids to train ids  
            
        except: 
            print('**************') 
            print(index)
            index = index - 1 if index > 0 else index + 1 
            return self.__getitem__(index) 
        
        return image, label, name


## Todo 
## now have to transfroms the dataset and then load model to train it have to same for acdc as well###
################


def city_train_data(cfg): 
    train_env = cfg['cityscapes']  
    cfg.train_data_dir = train_env.data_dir
    cfg.train_data_list = train_env.data_list

    # trainloader = data.DataLoader(
    #         cityscapesloader(cfg, cfg.train_data_dir, cfg.train_data_list, cfg.num_class, ignore_label=255, set='train', train_transform=train_transform),
    #         batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.worker, pin_memory=True) 
    
    city_train_data = cityscapesloader(cfg, cfg.train_data_dir, cfg.train_data_list, cfg.num_class, ignore_label=255, set='train', train_transform=train_transform)

    # return trainloader 
    return city_train_data

def city_val_data(cfg): 
    val_env = cfg['cityscapes_val'] 
    cfg.val_data_dir = val_env.data_dir
    cfg.val_data_list = val_env.data_list  

    # valloader = data.DataLoader(
    #         cityscapesloader(cfg, cfg.val_data_dir, cfg.val_data_list, cfg.num_class, ignore_label=255, set='val', val_transform=val_transform),
    #         batch_size=1, shuffle=True, num_workers=cfg.worker, pin_memory=True) 

    city_val_data = cityscapesloader(cfg, cfg.val_data_dir, cfg.val_data_list, cfg.num_class, ignore_label=255, set='val', val_transform=val_transform)

    # return valloader
    return city_val_data


    
        