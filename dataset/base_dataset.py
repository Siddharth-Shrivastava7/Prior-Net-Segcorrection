import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import torchvision
from torch.utils import data
from PIL import Image
import torchvision.transforms.functional as TF
import torch
import imageio
from . import transforms
from . import joint_transforms
from torch.nn import functional as F 
import random
from torchvision import transforms 
# from .network import dannet_pred
# from .network.dannet_pred import pred

## one caveat with TF.to_tensor...that it convert the value of the image between 0 and 1 


class BaseDataSet(data.Dataset):
    def __init__(self, cfg, root, list_path,dataset, num_class,  joint_transform=None, transform=None, label_transform = None, max_iters=None, ignore_label=255, set='val', plabel_path=None, max_prop=None, selected=None,centroid=None, wei_path=None):
        
        self.root = root
        self.list_path = list_path
        self.ignore_label = ignore_label
        self.set = set
        self.dataset = dataset
        self.transform = transform
        self.joint_transform = joint_transform
        self.label_transform = label_transform
        self.plabel_path = plabel_path
        self.centroid = centroid
        self.cfg = cfg

        # print(')))))')

        # if self.set !='train':
        #     self.list_path = (self.list_path).replace('train', self.set)

        self.img_ids =[]
        if selected is not None:
            self.img_ids = selected
        else:
            with open(self.list_path) as f:
                for item in f.readlines():
                    fields = item.strip().split('\t')[0]
                    if ' ' in fields:
                        fields = fields.split(' ')[0]
                    self.img_ids.append(fields)

        if not max_iters==None:
            # print(len(self.img_ids))
            # self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids))) # org
            # print(len(self.img_ids))
            pass

        # elif max_prop is not None:
        #     total = len(self.img_ids)
        #     to_sel = int(np.floor(total * max_prop))
        #     index = list( np.random.choice(total, to_sel, replace=False) )
        #     self.img_ids = [self.img_ids[i] for i in index]

        self.files = []
        self.id2train = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                          19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                          26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        if self.dataset =='synthia':
            imageio.plugins.freeimage.download()

        if dataset=='gta5':
            if self.plabel_path is None:
                label_root = osp.join(self.root, 'labels')
            else:
                label_root = self.plabel_path

            for name in self.img_ids:
                img_file = osp.join(self.root, "images/%s" % name)
                label_file = osp.join(label_root, "%s" % name)
                self.files.append({
                    "img": img_file,
                    "label": label_file,
                    "name": name
                })

        elif dataset=='cityscapes' or dataset=='cityscapes_val':
            if self.plabel_path is None:
                label_root = osp.join(self.root, 'gtFine', self.set)
            else:
                label_root = self.plabel_path 
            for name in self.img_ids:
                # print(name)
                nm = name.split('/')[-1] 
                # img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name)) # original
                img_file = osp.join(self.root, "leftImg8bit/%s/%s/%s" % (self.set, 'dark_city', nm))
                label_name = name.replace('leftImg8bit', 'gtFine_labelIds')
                label_file =osp.join(label_root, '%s' % (label_name))
                self.files.append({
                    "img": img_file,
                    "label":label_file,
                    "name": name
                })
                # print(img_file)
                # print(label_file)

        elif dataset=='rf_city' or dataset=='rf_city_val':
            # print('hi')
            for name in self.img_ids:
                # print(name)
                img_file = osp.join(self.root,  name)
                if 'gta_pred' in img_file:
                    name = name.split('/')[-1].replace('leftImg8bit','gtFine_color')
                    label_file = osp.join(self.root, 'rf_fake', name)
                else: 
                    name = name.split('/')[-1] 
                    label_file = osp.join(self.root, 'rf_real', name)
                self.files.append({
                    "img": img_file,
                    "label":label_file,
                    "name": name
                })
                # print(img_file)
                # print(label_file)

        elif dataset=='rf_city_dark' or dataset=='rf_city_dark_val':
            # print('hi')
            for name in self.img_ids:
                # print(name)
                img_file = osp.join(self.root,  name)
                if 'dark_city' in img_file:
                    # print(name)
                    name = name.split('/')[-1].replace('leftImg8bit','gtFine_color')
                    label_file = osp.join(self.root, 'rf_fake_dark', name)
                else: 
                    name = name.split('/')[-1].replace('leftImg8bit','gtFine_color')
                    label_file = osp.join(self.root, 'rf_real_dark', name)
                self.files.append({
                    "img": img_file,
                    "label":label_file,
                    "name": name
                })
                # print(img_file)
                # print('***************************')
                # print(label_file)

        elif dataset == 'darkzurich_val':
            if self.plabel_path is None:
                label_root = osp.join(self.root, 'gt')
            else:
                label_root = self.plabel_path 
            for name in self.img_ids:
                # print(name)
                img_file = osp.join(self.root, "rgb_anon/%s_rgb_anon.png" % (name))
                label_file =osp.join(label_root, '%s_gt_labelIds.png' % (name))

                # print(img_file)
                # print(label_file)

                self.files.append({
                    "img": img_file,
                    "label":label_file,
                    "name": name
                })
        
        elif dataset == 'dark_zurich_val_rf':
            for name in self.img_ids:
                # print(name)
                img_file =osp.join(self.root, name)
                if 'mgcda_pred' in name: 
                    nm = name.split('/')[-1].replace('rgb_anon','gt_labelColor')
                    label_file = osp.join(self.root, 'rf_fake', nm)
                else: 
                    nm = name.split('/')[-1]
                    label_file = osp.join(self.root, 'rf_real', nm)
                # print(img_file)
                # print(label_file)

                self.files.append({
                    "img": img_file,
                    "label":label_file,
                    "name": name
                })

        elif dataset == 'darkzurich':
            for name in self.img_ids:
                img_file = osp.join(self.root, "rgb_anon/%s_rgb_anon.png" % (name))
                label_root = '/home/sidd_s/scratch/dataset/dark_zurich_val/gt'
                if self.cfg.uncertain: 
                    label_file =osp.join(label_root, '%s_gt_label_valid_TrainIds.png' % (name)) 
                    # label_file =osp.join(label_root, '%s_gt_label_inv_TrainIds.png' % (name))  # here ignoring the uncertain regions 
                else: 
                    label_file =osp.join(label_root, '%s_gt_labelTrainIds.png' % (name))   ## original     
                self.files.append({
                    "img": img_file,
                    "label":label_file,
                    "name": name
                })

        elif dataset=='night_city':
            if self.plabel_path is None:
                label_root = self.root 
            else:
                label_root = self.plabel_path 
            for name in self.img_ids:
                img_file = osp.join(self.root, "%s" % (name))

                label_name = name.replace('_leftImg8bit', '_gtCoarse_labelIds')
                label_name = label_name.replace('leftImg8bit', 'gtCoarse_daytime_trainvaltest')
                label_file =osp.join(label_root, '%s' % (label_name))

                # print(img_file)
                # print(label_file)
                # print(name)

                self.files.append({
                    "img": img_file,
                    "label":label_file,
                    "name": name
                }) 
        elif dataset == 'acdc':
            # print('yo')
            for name in self.img_ids:
                # print(name)
                img_file = osp.join(self.root,  name)
                tup = (('acdc_trainval','acdc_gt'),('_rgb_anon.png','_gt_labelIds.png')) 
                for r in tup: 
                    lbname = name.replace(*r)
                lbname = lbname.replace('rgb_anon','gt') 
                label_file = osp.join(self.root, lbname)
                self.files.append({
                    "img": img_file,
                    "label":label_file,
                    "name": name
                })
                # print(label_file)
                # print(name)
                # print(img_file)
                # break

        elif dataset=='acdc_train_rf' or dataset=='acdc_val_rf':
            # print('************************') 
            for name in self.img_ids:
                # print(name)
                img_file = osp.join(self.root, name)
                # print('^^^^^^^')
                # print(img_file)
                # print(name)
                if 'pred_dannet' in img_file: 
                    # print('*****')
                    nm = name.split('/')[-1].replace('rgb_anon_color', 'gt_labelColor')
                    fk_save = 'acdc_gt/rf_fake_dannet_' + self.set 
                    label_file = osp.join(self.root, fk_save, nm)
                    # print(label_file)
                # else:
                #     # print('>>>>>')
                #     nm = name.split('/')[-1] 
                #     re_save = 'acdc_gt/rf_real_dannet_' + self.set 
                #     label_file = osp.join(self.root, re_save, nm) 
                #     # print(label_file)
                    self.files.append({
                        "img": img_file,
                        "label":label_file,
                        "name": name
                    })
                # print(img_file)
                # print('************')
                # print(label_file) 
                # break  

        elif dataset=='acdc_train_rf_tensor' or dataset=='acdc_val_rf_tensor':
        # print('************************') 
            for name in self.img_ids:
                # print(name)
                # img_file = osp.join(self.root, name)
                # print('^^^^^^^')
                # print(img_file)
                # print(name)
                # print('*****')
                nm = name.split('/')[-1].replace('.png','.pt')
                fk_save = 'acdc/tensor_' + self.set + '_pred'
                img_file = osp.join(self.root, fk_save, nm) 
                nm = name.split('/')[-1].replace('rgb_anon', 'gt_labelColor')
                fk_save = 'acdc_gt/rf_fake_dannet_' + self.set 
                root = '/home/sidd_s/scratch/data'
                label_file = osp.join(root, fk_save, nm) 
                # print(label_file)
                self.files.append({
                    "img": img_file,
                    "label":label_file,
                    "name": name
                })
                # print(img_file)
                # print('************')
                # print(label_file) 
                # break

        elif dataset == 'acdc_dz_val_rf_tensor':
            for name in self.img_ids:
                # print(name)
                if 'dannet_pred' in name:
                    nm = name.split('/')[-1].replace('rgb_anon_color.png', 'rgb_anon.pt')
                    fk_save = '/home/sidd_s/scratch/saved_models_hpc/saved_models/DANNet/dz_val/tensor_pred'
                    img_file = osp.join(fk_save, nm) 
                    nm = name.split('/')[-1].replace('rgb_anon_color', 'gt_labelColor')
                    fk_save = 'rf_fake_dannet'    ## original  
                    # print('*****')
                    # fk_save = 'rf_fake_dannet_variation' ## change just for using the variation map 
                    # fk_save = 'rf_fake_dannet_entropy' # mod for entrpy map inclusion
                    label_file = osp.join(self.root, fk_save, nm)  
                    # print('*****')
                    self.files.append({
                        "img": img_file,
                        "label":label_file,
                        "name": name
                    })
                    # print(img_file)
                    # print('************')
                    # print(label_file) 
                    # break  
        
        elif dataset == 'acdc_dz_val_rf':
            for name in self.img_ids:
                # print(name)
                img_file = osp.join(self.root, name)
                if 'dannet_pred' in img_file:
                    nm = name.split('/')[-1].replace('rgb_anon_color', 'gt_labelColor')
                    fk_save = 'rf_fake_dannet'    
                    label_file = osp.join(self.root, fk_save, nm) 
                    # print(nm)
                else: 
                    # print('***********')
                    nm = name.split('/')[-1] 
                    # print(nm)  
                    re_save = 'rf_real_dannet'    
                    label_file = osp.join(self.root, re_save, nm)  
                
                self.files.append({
                    "img": img_file,
                    "label":label_file,
                    "name": name
                })

        elif dataset == 'acdc_dz_val_rf_vr':
            for name in self.img_ids:
                # print(name)
                if 'dannet_pred' in name:
                    nm = name.split('/')[-1].replace('rgb_anon_color.png', 'rgb_anon.png')
                    fk_save = '/home/cse/phd/anz208849/scratch/saved_models/DANNet/dz_val/seg_variation_map_bin'
                    img_file = osp.join(fk_save, nm) 
                    nm = name.split('/')[-1].replace('rgb_anon_color', 'gt_labelColor')
                    fk_save = 'rf_fake_dannet'    
                    label_file = osp.join(self.root, fk_save, nm)  
                    self.files.append({
                        "img": img_file,
                        "label":label_file,
                        "name": name
                    })
                    # print(img_file)
                    # print('*****************')
                    # print(label_file)
                # print(label_file) 
                    # print(label_file)
                # print(label_file) 
                    # print(label_file)

        elif dataset == 'acdc_train_tensor' or dataset == 'acdc_val_tensor':
            # print('*************************')
            # print(self.set)
            for name in self.img_ids:
                nm = name.split('/')[-1].replace('.png','.pt')
                pred_save = 'acdc/tensor_' + self.set + '_pred'
                img_file = osp.join(self.root, pred_save, nm) 

                # nm = name.split('/')[-1].replace('rgb_anon', 'gt_labelColor').replace('acdc_trainval','acdc_gt')
                replace = (("_rgb_anon", "_gt_labelTrainIds"), ("acdc_trainval", "acdc_gt"), ("rgb_anon", "gt"), (".pt", ".png"))
                nm = name
                for r in replace: 
                    nm = nm.replace(*r)  
                root = '/home/sidd_s/scratch/data_hpc/data'
                label_file = osp.join(root, nm) 
                # print(label_file)
                # print('***************')
                # print(img_file)
                # break
                # print(label_file)
                self.files.append({
                    "img": img_file,
                    "label":label_file,
                    "name": name
                })

        elif dataset == 'acdc_train_label' or dataset == 'acdc_val_label':
            for name in self.img_ids:
                img_file = osp.join(self.root, name)

                replace = (("_rgb_anon", "_gt_labelTrainIds"), ("acdc_trainval", "acdc_gt"), ("rgb_anon", "gt"))
                nm = name
                for r in replace: 
                    nm = nm.replace(*r)  
                if self.cfg.uncertain:
                    nm = nm.replace('_gt_labelTrainIds', '_gt_label_validinv_TrainIds') 
                label_file = osp.join(self.root, nm) 
                # print(label_file)
                self.files.append({
                    "img": img_file,
                    "label":label_file,
                    "name": name
                }) 
        
        elif dataset == 'dz_val_tensor':
            for name in self.img_ids:
                if 'dannet_pred' not in name:
                    nm = name.split('/')[-1].replace('_gt_labelColor.png', '_rgb_anon.pt')
                    img_save = '/home/sidd_s/scratch/saved_models_hpc/saved_models/DANNet/dz_val/tensor_pred'
                    img_file = osp.join(img_save, nm)
                    nm = name.replace('_gt_labelColor.png','_gt_labelTrainIds.png')   
                    label_file = osp.join(self.root, nm)
                    # print(label_file)
                    # print('***************')
                    # print(img_file)
                    # break
                    self.files.append({
                    "img": img_file,
                    "label":label_file,
                    "name": name
                    })

        elif dataset in ['city_dark_img_tensor', 'city_dark_img_tensor_val']:
            for name in self.img_ids:
                img_file = osp.join(self.root, 'black_out_' + self.set , name)
                label_file = osp.join(self.root, 'gtFine', self.set, name.replace('_leftImg8bit', '_gtFine_labelIds'))
                # print(label_file)
                # print('***************')
                # print(img_file)
                # break
                self.files.append({
                    "img": img_file, 
                    "label": label_file, 
                    "name": name
                })
        
        elif dataset in ['acdc_city_dark', 'acdc_city_dark_val']:
            for name in self.img_ids:
                img_file = osp.join(self.root, self.set, 'img', name)
                label_file = osp.join(self.root, self.set, 'gt', name.replace('_leftImg8bit.png','_gtFine_labelIds.png'))
                self.files.append({
                    "img": img_file, 
                    "label": label_file, 
                    "name": name
                })
        
        elif dataset in ['acdc_fake_ce_train', 'acdc_fake_ce_val']:
            for name in self.img_ids:
                img_file = osp.join(self.root, name)
                nm = name.split('/')[-1].replace('_rgb_anon.png', '_gt_labelTrainIds.png') 
                label_path = '/home/sidd_s/scratch/data/acdc_gt/acdc_fake_ce_' + self.set
                label_file = osp.join(label_path, nm)
                self.files.append(
                    {
                        "img": img_file,
                        "label": label_file,
                        "name": name
                    }
                )

        elif dataset in ['acdc_fake_ce_real_en_train', 'acdc_fake_ce_real_en_val']:
            for name in self.img_ids:
                img_file = osp.join(self.root, name)
                nm = name.split('/')[-1].replace('_rgb_anon.png', '_gt_labelTrainIds.png') 
                label_path = '/home/sidd_s/scratch/data/acdc_gt/acdc_fake_ce_real_en_' + self.set
                label_file = osp.join(label_path, nm)
                self.files.append(
                    {
                        "img": img_file,
                        "label": label_file,
                        "name": name
                    }
                )             

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        name = datafiles["name"]
        # print(datafiles["img"])
        # print('**********')
        # print(datafiles["label"])
        try:
            # image = Image.open(datafiles["img"]).convert('RGB')
            if self.dataset in ['acdc_val_rf_tensor', 'acdc_train_rf_tensor', 'acdc_dz_val_rf_tensor']:
                if self.set == 'val': 
                    # x = []
                    # image = torch.tensor(torch.load(datafiles["img"]))
                    # image = image.transpose(2,0,1)

                    # label = Image.open(datafiles['label'])
                    # # how to resize have to seee  and don't crop the label rather resize only the image..cause it will good in evaluation
                    # # ya so...no rcrop in val...only resizing and in a way calculating by original 
                    # # i, j, h, w = transforms.RandomCrop.get_params(                  # 512 x 512 cropping is in orginal 
                    # #                 label, output_size=(512, 512)) 
                    # # label = TF.crop(label, i, j, h,w)  
                    # # label = TF.resize(label, (512, 512), interpolation = Image.NEAREST) # commenting for original size evalutaion  

                    # label = np.array(label, dtype = np.int32)   
                    # label[label == 127] = 1
                    # # label = TF.to_tensor(label).to(dtype=torch.uint8) ## wrong way....
                    # label = torch.tensor(label)
                    # label = label.squeeze(dim=0) 
                    # # print(torch.unique(label)) 
                    # # print('***********')

                    # for ch in image: 
                    #     ch = Image.fromarray(ch)
                    #     # ch = TF.crop(ch, i,j ,h,w) 
                    #     # ch = TF.resize(ch, (512, 512)) # see this # (512,512) for unet mod and unet gan goes for (256,256)
                    #     ch = TF.resize(ch, (512, 512), interpolation = Image.NEAREST)     
                    #     # x.append(TF.to_tensor(ch)) ## wrong way....
                    #     x.append(torch.tensor(np.array(ch)))
                    # # image = torch.cat(x) ## wrong way.... 
                    # image = torch.stack(x, dim = 0)  

                    # image = F.softmax(image, dim = 0) ##an experiment to not use..use it
                    # # print(image.shape)
                    # # print('**********')

                    ### modified one....for val
                    image = torch.tensor(torch.load(datafiles["img"]))
                    image = image.transpose(2,0).transpose(1,2) 

                    label = Image.open(datafiles['label'])
                    label = np.array(label, dtype = np.int32)   
                    label[label == 127] = 1
                    label = torch.tensor(label)
                    label = label.squeeze(dim=0) 
         
                    ## pytorch transfroms ... how to compose them effectively 
                    # img_transforms_compose = transforms.Compose([transforms.ToPILImage(), transforms.Resize((512,512),interpolation=Image.NEAREST]) ## not using the random crop (now)
                    # img_trans = img_transforms_compose(image)  
                    
                else:
                    # seed = np.random.randint(2147483647) 
                    # x = []
                    # # 19 channel image transformation                
                    # name = datafiles["name"] 
                    # image = torch.load(datafiles["img"])
                    # image = image.transpose(2,0,1)
                    
                    # label = Image.open(datafiles['label'])
                    # # i, j, h, w = transforms.RandomCrop.get_params(
                    # #                 label, output_size=(512, 512))
                    # # print('****************')
                    # # print(i,j,h,w)
                    # tfms = transforms.Compose([
                    #         transforms.RandomHorizontalFlip(),
                    #         transforms.RandomVerticalFlip()])
                    # for ch in image: 
                    #     random.seed(seed) 
                    #     torch.manual_seed(seed)
                    #     ch = Image.fromarray(ch)
                    #     # ch = TF.crop(ch, i,j ,h,w)
                    #     ch = TF.resize(ch, (512, 512), interpolation = Image.NEAREST)  
                    #     ch = tfms(ch)
                    #     # x.append(TF.to_tensor(ch)) ## wrong way....
                    #     x.append(torch.tensor(np.array(ch)))
                    # # image = torch.cat(x) ## wrong way...
                    # image = torch.stack(x, dim = 0)
                    # # image = F.softmax(image, dim = 0) ## an experiment to not use...use it 
                    # random.seed(seed) 
                    # torch.manual_seed(seed)
                    # # label = TF.crop(label, i, j, h,w)
                    # label = TF.resize(label, (512, 512), interpolation = Image.NEAREST)  
                    # label = tfms(label)
                    # label = np.array(label, dtype = np.int32)
                    # label[label == 127] = 1
                    # # label = TF.to_tensor(label).to(dtype=torch.uint8) ## wrong way....
                    # label = torch.tensor(label)
                    # label = label.squeeze(dim=0) 
                
                    # # print(torch.unique(label)) #tensor([0, 1, 255], dtype=torch.int32)
                    # # print('####################')
                    # # print(label.shape) #torch.Size([1080, 1920]); torch.Size([256, 256])
                    # # print(image.shape) # torch.Size([19, 1080, 1920]); torch.Size([19, 256, 256])
                    # # print('&&&&&&&&&&&&&&&&&&&&')
                    # # print(torch.is_tensor(image)) 
                    # # print(torch.is_tensor(label))   
                    # # print('yo')

                    ### modified one...for train 
                    ## only augmentation currently, doing is the random crop

                    x = []
                    image = torch.load(datafiles["img"]) 
                    image = image.transpose(2,0,1) 

                    label = Image.open(datafiles['label']) 
                    i, j, h, w = transforms.RandomCrop.get_params(
                                    label, output_size=(512, 512))
                
                    for ch in image:
                        ch = Image.fromarray(ch) 
                        # ch = TF.resize(ch, (512, 512), interpolation = Image.NEAREST)  
                        ch = TF.crop(ch, i,j ,h,w)
                        x.append(torch.tensor(np.array(ch))) 
                    image = torch.stack(x, dim = 0)

                    label = TF.crop(label, i, j, h,w) 
                    # label = TF.resize(label, (512, 512), interpolation = Image.NEAREST)
                    label = np.array(label, dtype = np.int32)
                    label[label == 127] = 1 
                    label = torch.tensor(label)
                    label = label.squeeze(dim=0)

                    ### modified one...for train


            # if self.dataset == 'rf_city' or self.dataset == 'rf_city_val':
            #     # print('&???*******?')
            #     im = np.array(image).shape
            #     if 'gta' in datafiles["name"]:
            #         # print('hi')
            #         label = np.zeros((im[0], im[1]), dtype = np.long) #fake label # for cross entropy
            #         # label = np.zeros((im[0], im[1]), dtype = np.uint8) #fake label 
            #     else:
            #         im_arr = np.array(image)
            #         indices = np.where(np.all(im_arr == (0,0,0), axis=-1))
            #         black_inx = np.transpose(indices)
            #         label = np.ones((im[0], im[1]), dtype = np.long) # real label # for cross entropy
            #         label[black_inx[:,0], black_inx[:,1]] = 255
            #         # label = np.ones((im[0], im[1]), dtype = np.uint8) # real label

            #     label = Image.fromarray(label.astype(np.uint8))
                
            # elif self.dataset == 'darkzurich' and self.plabel_path is None: # trg no gt labels
            #     # print(self.dataset)
            #     image = Image.open(datafiles["img"]).convert('RGB')
            #     label = []
            
            ## tensor to gt label (for end to end learning)
            elif self.dataset == 'acdc_train_tensor' or self.dataset == 'acdc_val_tensor' or self.dataset=='dz_val_tensor':
                if self.set == 'val': 
                    # print('&&&&&&&&&&&')
                    # x = [] 
                    # image = torch.load(datafiles["img"])
                    # image = image.transpose(2,0,1)
                    # image = torch.tensor(image)
                    # # print(image.shape)  ## (19, 1080, 1920)  
                    # # for ch in image:  
                    # #     ch = Image.fromarray(ch)
                    # #     # ch = TF.resize(ch, (512, 512), interpolation = Image.NEAREST)
                    # #     # x.append(TF.to_tensor(ch))  ## wrong way....
                    # #     x.append(torch.tensor(np.array(ch)))
                    # # image = torch.cat(x) ## wrong way...
                    # # image = torch.stack(x, dim = 0)
                    # # print(image.shape) # torch.Size([19, 1080, 1920])
                    # # image = F.softmax(image, dim = 0)  ## softmax ...making it a proba distribution...not using it now 
                    # # print(torch.unique(image))
                    # # print(image)

                    # label = Image.open(datafiles['label']) 
                    # # print(datafiles['label']) ## its correct 
                    # label = torch.tensor(np.array(label))
                    # # label = TF.to_tensor(label).to(dtype=torch.uint8)  #### this is shit....it convert the tensor into 0 and 1 ....yaar...alright i will try ## wrong way....
                    # label = label.squeeze(dim=0)  
                    # print(label.shape) # torch.Size([1080, 1920])
                    # print(torch.unique(label)) ## tensor([0, 1], its learning wrong....ohhhh gooooddd
                    ## original size eval...let see...if its tooo slow then resize the image here ...cause now using cross entropy loss of pytorch not nll...
                
                    ### modified...val 
                    image = torch.tensor(torch.load(datafiles["img"])) 
                    image = image.transpose(2,0).transpose(1,2) ## original
                    
                    label = Image.open(datafiles['label'])
                    label = torch.tensor(np.array(label))
                    label = label.squeeze(dim=0) 
                    ### modified...val
                
                else: 
                    # # print('************')
                    # x = []
                    # seed = np.random.randint(2147483647) 
                    # image = torch.load(datafiles["img"])
                    # image = image.transpose(2,0,1)
                    # # tfms = transforms.Compose([  ### testing once commenting it 
                    # #         transforms.RandomHorizontalFlip(),
                    # #         transforms.RandomVerticalFlip()])
                    
                    # label = Image.open(datafiles['label'])
                    # i, j, h, w = transforms.RandomCrop.get_params(
                    #                 label, output_size=(512, 512)) 

                    # for ch in image: 
                    #     random.seed(seed) 
                    #     torch.manual_seed(seed)
                    #     ch = Image.fromarray(ch)
                    #     # ch = TF.resize(ch, (512, 512), interpolation = Image.NEAREST)  ## different method of reszeing see...or first verify this 
                    #     # ch = TF.crop(ch, i,j ,h,w) 
                    #     # ch = tfms(ch)
                    #     # x.append(TF.to_tensor(ch)) ## wrong way....
                    #     x.append(torch.tensor(np.array(ch))) 
                    # # image = torch.cat(x)## wrong way...
                    # image = torch.stack(x, dim = 0)
                    # # image = F.softmax(image, dim = 0) ## softmax ...making it a proba distribution ### let it be..don't use this...cause now using cross entropy loss of pytorch not nll...

                    # random.seed(seed) 
                    # torch.manual_seed(seed)
                    # # label = TF.crop(label, i, j, h, w)
                    # label = TF.resize(label, (512, 512), interpolation = Image.NEAREST)  
                    # # label = tfms(label)
                    # # label = TF.to_tensor(label).to(dtype=torch.uint8) ## wrong way....
                    # label = torch.tensor(np.array(label))
                    # label = label.squeeze(dim=0)

                    ### Modified for train.... only cropping...not resizing will do if possible on the next iterations...
                    x = [] 
                    image = torch.load(datafiles["img"]) 
                    image = image.transpose(2,0,1)
                    label = Image.open(datafiles['label']) 
                    # i, j, h, w = transforms.RandomCrop.get_params(
                    #                 label, output_size=(512, 512))
                
                    for ch in image:
                        ch = Image.fromarray(ch) 
                        ch = TF.resize(ch, (512, 512), interpolation = Image.NEAREST)  
                        # ch = TF.crop(ch, i,j ,h,w)
                        x.append(torch.tensor(np.array(ch))) 
                    image = torch.stack(x, dim = 0)

                    # label = TF.crop(label, i, j, h, w) 
                    label = TF.resize(label, (512, 512), interpolation = Image.NEAREST)
                    label = torch.tensor(np.array(label))
                    label = label.squeeze(dim=0)
                    ### Modified for train.... 



            # elif self.dataset == 'acdc_train_rf' or self.dataset == 'acdc_val_rf' or self.dataset == 'rf_city' or self.dataset == 'rf_city_val' or self.dataset == 'rf_city_dark' or self.dataset=='rf_city_dark_val' or self.dataset == 'dark_zurich_val_rf' or self.dataset=='acdc_dz_val_rf':
            elif self.dataset in ['acdc_train_rf', 'acdc_val_rf', 'rf_city', 'rf_city_val', 
                'rf_city_dark', 'rf_city_dark_val', 'dark_zurich_val_rf', 'acdc_dz_val_rf']:  
                # print('*************>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>') 
                # print(self.dataset)
                image = Image.open(datafiles["img"]).convert('RGB')
                label = np.array(Image.open(datafiles['label']), dtype = np.int32)
                label[label == 127] = 1
                label = Image.fromarray(label.astype(np.uint8)) 
                if self.joint_transform is not None:
                    image, label = self.joint_transform(image, label, None)
                if self.label_transform is not None:
                    label = self.label_transform(label)
                if self.transform is not None:
                    image = self.transform(image)

            elif self.dataset == 'acdc_dz_val_rf_vr':
                # print('*****')
                # print(datafiles["img"])
                image = np.array(Image.open(datafiles["img"]), dtype = np.int32)
                # print('&&&&&&&')
                # print('***************')
                # print(image.shape)
                image[image==0] = 1
                image[image==255] = 0
                # print(np.unique(image))
                image = torch.from_numpy(image)
                label = np.array(Image.open(datafiles['label']), dtype = np.int32)
                label[label == 127] = 1
                # print(np.unique(label))
                label = Image.fromarray(label.astype(np.uint8))
                # label_transform = transforms.MaskToTensor()
                # label = label_transform(label)
                if self.joint_transform is not None:
                    image, label = self.joint_transform(image, label, None)
                if self.label_transform is not None:
                    label = self.label_transform(label)
                if self.transform is not None:
                    image = self.transform(image)
            
            elif self.dataset in ['acdc_train_label', 'acdc_val_label', 'acdc_fake_ce_val', 'acdc_fake_ce_train', 'acdc_city_dark', 'acdc_city_dark_val', 'darkzurich', 'acdc_fake_ce_real_en_train', 'acdc_fake_ce_real_en_val']:  ## check this, once ..........................
                
                mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  ## image net mean and std 

                if self.set=='val':
                    
                    transforms_compose_img = transforms.Compose([
                        transforms.Resize((540, 960)),
                        # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), ## added for aug2 exp (only this in aug3) ## use only in training 
                        transforms.ToTensor(),
                        transforms.Normalize(*mean_std), ## use only in training 
                        # transforms.RandomErasing() ## added for aug2 exp 
                    ])
                    image = Image.open(datafiles["img"]).convert('RGB') 
                    image = transforms_compose_img(image)    
                    image = torch.tensor(np.array(image)).float() 
                    # image = image.transpose(2,0).transpose(1,2) ## not required when using transforms 

                    # print('*****************************')

                    transforms_compose_label = transforms.Compose([
                        transforms.Resize((1080,1920), interpolation=Image.NEAREST)])
                    # print('*****************************') 
                    label = Image.open(datafiles["label"]) 
                    label = transforms_compose_label(label)

                    if self.dataset in ['acdc_city_dark_val']:
                        label = np.array(label, dtype=np.uint8)
                        label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
                        for k, v in self.id2train.items():
                            label_copy[label == k] = v
                        label = torch.tensor(np.array(label_copy))
                    else:
                        label = torch.tensor(np.array(label)) 
                    # print(label.shape)
                
                else:
                    image = Image.open(datafiles["img"]).convert('RGB') 
                    # transforms_compose_img = transforms.Compose([transforms.Resize((512,512)), transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2), transforms.ToTensor(), transforms.Normalize(*mean_std)])  

                    transforms_compose_img = transforms.Compose([transforms.Resize((512,512)), transforms.ToTensor(), transforms.Normalize(*mean_std)])   ## no random gittering in exp3 models...performing better than the 

                    img_trans = transforms_compose_img(image) 
                    # print(img_trans)
                    image = torch.tensor(np.array(img_trans)).float()
                    # print(image.shape)
                    # image = image.transpose(2,0).transpose(1,2)  ## not required when using transforms 
                    # print(image.shape)  #torch.Size([3, 512, 512])   
                    # print('*******')

                    transforms_compose_label = transforms.Compose([transforms.Resize((512,512),interpolation=Image.NEAREST)])

                    label = Image.open(datafiles["label"]) 
                    label = transforms_compose_label(label)

                    if self.dataset in ['acdc_city_dark']:
                        label = np.array(label, dtype=np.uint8)
                        label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
                        for k, v in self.id2train.items():
                            label_copy[label == k] = v
                        label = torch.tensor(np.array(label_copy))
                    else:
                        label = torch.tensor(np.array(label)) 
                    # print(label.shape) # torch.Size([512, 512]) 

            elif self.dataset in ['city_dark_img_tensor', 'city_dark_img_tensor_val']:
                if self.set=='val':
                    transforms_compose = transforms.Compose([transforms.Resize((1080,1920),interpolation=Image.NEAREST)])
                    
                    image = Image.open(datafiles["img"]).convert('RGB') 
                    image = transforms_compose(image)   
                    image = torch.tensor(np.array(image)).float()
                    image = image.transpose(2,0).transpose(1,2)  
                      
                    
                    # label = np.array(Image.open(datafiles["label"]), dtype = np.uint8) 
                    label = Image.open(datafiles["label"]) 
                    label = transforms_compose(label) 
                    label = np.array(label, dtype=np.uint8)
                    label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
                    for k, v in self.id2train.items():
                        label_copy[label == k] = v
                    label = torch.tensor(np.array(label_copy)) 
                else: 
                    # print('**************')
                    image = Image.open(datafiles["img"]).convert('RGB') 
                    transforms_compose = transforms.Compose([transforms.Resize((512,512),interpolation=Image.NEAREST), transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)])
                    # transforms_compose_img = transforms.Compose([transforms.Resize((512,512),interpolation=Image.NEAREST), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])  ## resize, normlising
                    # img_trans = transforms_compose_img(image) 
                    img_trans = transforms_compose(image) 
                    # print('**************')
                    # print(img_trans)
                    # print(img_trans.shape)  # torch.Size([3, 512, 512])
                    image = torch.tensor(np.array(img_trans)).float()
                    image = image.transpose(2,0).transpose(1,2)  
                    # image = img_trans
                    # print(image.shape) # torch.Size([3, 512, 512])
                    # print('**************')
                    # print(torch.unique(image))
                    
                    # transforms_compose_label = transforms.Compose([transforms.Resize((512,512),interpolation=Image.NEAREST)])
                    label = Image.open(datafiles["label"]) 
                    # print(np.array(label).shape)
                    # print(np.max(label))
                    # label_trans = transforms_compose_label(label) 
                    label_trans = transforms_compose(label) 
                    # print(np.array(label_trans).shape) # (512, 512)
                    # print(np.max(label_trans)) 
                    # print(np.unique(label_trans))
                    # print(np.array(label_trans))
                    label_trans = np.array(label_trans, dtype = np.uint8) 
                    label_copy = 255 * np.ones(label_trans.shape, dtype=np.uint8) 
                    # print('**************')  
                    for k, v in self.id2train.items():
                        # print(k,v)
                        # print(label_trans==k)
                        label_copy[label_trans == k] = v
                        # print(label_copy[label_trans == k]) 
                    # print('**************')  
                    label = torch.tensor(np.array(label_copy))
                    # print(torch.unique(label))   
                    # print(label.shape) # torch.Size([512, 512])

            else:
                image = Image.open(datafiles["img"]).convert('RGB')
                # print(self.dataset)
                label = Image.open(datafiles["label"])
                label = np.asarray(label, np.uint8)
                label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
                if self.plabel_path is None:
                    for k, v in self.id2train.items():
                        label_copy[label == k] = v
                else:
                    label_copy = label
                label = Image.fromarray(label_copy.astype(np.uint8))
                if self.joint_transform is not None:
                    image, label = self.joint_transform(image, label, None)
                if self.label_transform is not None:
                    label = self.label_transform(label)
                if self.transform is not None:
                    image = self.transform(image)


            # original >>>>>>                               #if image as the input
            # if self.joint_transform is not None:
            #     # print('*****************')
            #     image, label = self.joint_transform(image, label, None)
            # # print('>>>>>>>>>>>>>>>>')
            # if self.label_transform is not None:
            #     # print('&&&&&&&&&&&&&&&&&&&')
            #     label = self.label_transform(label)
            
            # name = datafiles["name"] 
            # # print(name)
            # # print('*****************')           
            # if self.transform is not None:
            #     image = self.transform(image)
            #     # image = np.asarray(image, np.float32)
            #     # print('>>>>>>>>>>>>>>')
            #     # image = image.transpose((2, 0, 1)) 
            #     # image = torch.from_numpy(image) 
            # original >>>>>>

        except Exception as e:
            # print('hi')
            print(index)
            index = index - 1 if index > 0 else index + 1
            return self.__getitem__(index)
        return image, label, 0, 0, name


