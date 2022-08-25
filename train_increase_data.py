## including night city data labels; to increase data variety and validating with acdc val; 
## train -> night city + acdc night 
## val -> acdc val 

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
from dataset.network.dannet_pred import pred    
from utils.optimize import * 
import  torch.optim as optim 
from tensorboardX import SummaryWriter 
from PIL import Image  
from dataset.network.dannet_pred import pred 
import network 
import argparse 
from torchsummary import summary 
import json  
import dataset.joint_transforms as joint_transforms  
import glob


parser = argparse.ArgumentParser() 

parser.add_argument("--gpu_id", type=str, default='6',
                        help="GPU ID") 
available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
available_models.append('unet')
parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16]) 
parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name') 
parser.add_argument("--num_classes", type=int, default=19,
                        help="num classes (default: 19)")  
parser.add_argument("--num_ip_channels", '-ip_chs',type=int, default=3,
                        help="num of input channels (default: 3)")  
parser.add_argument("--channel19", '-ch19',action='store_true', default=False,
                        help="to use colored o/p")   # may be req 
parser.add_argument("--restore_from", type=str, default='/home/sidd_s/scratch/saved_models/citydeeplabv3+/best_deeplabv3plus_mobilenet_cityscapes_os16.pth',
                        help='prior net trained model path')  
parser.add_argument("--multigpu", '-mul',action='store_true', default=False,
                        help="multigpu training") 
parser.add_argument("--eval_prior", '-ep', action='store_true', default=False,
                        help="evaluting the trained prior net model") # may be req 
parser.add_argument("--model_dannet", type=str, default='PSPNet',
                        help='seg model of dannet')     
parser.add_argument("--restore_from_da", type=str, default='/home/sidd_s/scratch/saved_models/DANNet/dannet_psp.pth',
                        help='Dannet pre-trained model path') 
parser.add_argument("--restore_light_path", type=str, default='/home/sidd_s/scratch/saved_models/DANNet/dannet_psp_light.pth',
                        help='Dannet pre-trained light net model path') 
parser.add_argument("--save_writer_name", '-svnm', type=str, default='1024_50n_100p_bt4', 
                        help='saving tensorboard file name')  # may be req  
parser.add_argument("--learning_rate", '-lr', type=float, default=3e-4, # may be req  
                        help="learning rate for trainig prior net") 
parser.add_argument("--momentum", '-m', type=float, default=0.9, 
                        help="momentum for sgd optimiser") 
parser.add_argument("--weight_decay", '-wd', type=float, default=0.0005, # increased weight decay L2 norm
                    help="weight_decay for sgd optimiser") 
parser.add_argument("--batch_size", type=int, default=4)  # may be req 
parser.add_argument("--worker", type=int, default=4)   
parser.add_argument("--print_freq", type=int, default=1)  
parser.add_argument("--epochs", type=int, default=500)   # may be req 
parser.add_argument("--snapshot", type=str, default='/home/sidd_s/scratch/saved_models/acdc/dannet',
                        help='model saving directory')                 
parser.add_argument("--synthetic_perturb", type=str, default='synthetic_manual_dannet_50n_100p_1024im', # manual color decided acc to seperablity b/w classes
                        help='evalutating technique')
parser.add_argument("--end_learning_rate", type=float, default=3e-4, # may be req  
                        help="end learning rate required in the lr scheduling") 
parser.add_argument("--power", type=float, default=0.9,  
                        help="power to be used in learning rate scheduling")    
parser.add_argument("--restore_from_color_mapping", type=str, default='None') 
parser.add_argument("--weighted_ce", '-wce',action='store_true', default=False,
                        help="multigpu training") 
parser.add_argument("--norm",action='store_true', default=False,
                        help="normalisation of the image")
parser.add_argument("--ignore_classes", nargs="+", default=[],  
                        help="ignoring classes...blacking out those classes") 
parser.add_argument("--img_size", type=int, default=1024,  # may be req 
                        help="size of image or label, which is to be trained")  
parser.add_argument("--naive_weighting", action='store_true', default=False,
                        help="naive weighting in ce loss") 
parser.add_argument("--dannet_acdc_weighting",action='store_true', default=False,
                        help="dannet type of acdc prediction weighting") 
parser.add_argument("--scheduler",action='store_true', default=False,
                        help="lr scheduling while training") 
parser.add_argument("--augment",action='store_true', default=False,
                        help="using augmentation during training") 
parser.add_argument("--small_model",action='store_true', default=False,
                        help="using augmentation during training")  ## denoising auto encoder (thinking of...let's seeee)
parser.add_argument("--mask_main_path", type=str, default='/home/sidd_s/scratch/dataset/random_bin_masks/',
                        help='random binary masks path for using in augmentation on the fly')
parser.add_argument("--random_bin_augment",action='store_true', default=False,
                        help="using augmentation during training")   
parser.add_argument("--dropout",action='store_true', default=False,
                        help="use of dropout in unet") 
args = parser.parse_args()

root_ac = '/home/sidd_s/scratch/dataset/acdc_trainval/rgb_anon/night/dannet_pred/' 
root_nc = '/home/sidd_s/scratch/dataset/night_city/dannet_pred/' 

## dataset init 
def init_train_data(args): 
    trainloader = data.DataLoader(
            Night_city_Acdc_night(args, root_ac, root_nc, set='train'),
            batch_size=args.batch_size, shuffle=True, num_workers=args.worker, pin_memory=True) 
    return trainloader 

## dataset init 
def init_val_data(args): 
    valloader = data.DataLoader(
            Night_city_Acdc_night(args, root_ac, root_nc, set='val'),
            batch_size=1, shuffle=True, num_workers=args.worker, pin_memory=True)
    return valloader

def perturb_gt_gen(mask_path, gt_path, pred_path):
    mask =  Image.open(mask_path).convert('L')
    gt = Image.open(gt_path)  
    pred = Image.open(pred_path)  
    if 'rgb_anon' in pred_path: 
        pred = pred.resize((1024, 512), Image.NEAREST) # for  making prediction of dannet for ac consistent
    gt = gt.resize((1024, 512), Image.NEAREST) # for making sure nc (& ac) dataset consistency shape{512,1024}
    gt = np.array(gt)
    pred = np.array(pred)
    mask = np.array(mask.resize((gt.shape[1], gt.shape[0]), Image.NEAREST)) # strange clouds of dimensions
    mask[mask==255] = 1 
    gt[mask==1] = pred[mask==1] 
    per_gt = Image.fromarray(gt)
    return per_gt


class Night_city_Acdc_night(data.Dataset): 
    def __init__(self, args, root_ac, root_nc, set):
        super().__init__()
        
        self.root_ac = root_ac
        self.root_nc = root_nc
        self.args = args 
        self.set = set 
        self.files = [] 
        self.img_ids = []
        self.mask_lst = os.listdir(self.args.mask_main_path) 
        self.per_gt_val_path_ac = '/home/sidd_s/scratch/dataset/acdc_trainval/rgb_anon/night/synthetic/val/synthetic_manual_dannet_random_size_shape/'
        
        # not req for now
        # self.per_gt_val_path_nc = '/home/sidd_s/scratch/dataset/night_city/synthetic_perturb_dannet_random_size_shape_color_val' 
        
        if self.set == 'train':
            # ac images
            self.search_img_ac = os.path.join(self.root_ac, self.set, '*_rgb_anon.png')
            filesimgac = glob.glob(self.search_img_ac)
            filesimgac.sort()  
            # nc images
            self.search_img_nc = os.path.join(self.root_nc, self.set, '*.png') 
            filesimgnc = glob.glob(self.search_img_nc) 
            filesimgnc.sort() 
            ## list of images (ac+nc) 
            self.img_ids = filesimgac + filesimgnc
        else: 
            # ac images
            self.search_img_ac = os.path.join(self.root_ac, self.set, '*_rgb_anon.png')
            filesimgac = glob.glob(self.search_img_ac)
            filesimgac.sort() 
            self.img_ids = filesimgac
            
        
        for img_file in self.img_ids:
            name = img_file.split('/')[-1]
            if 'acdc' in img_file: 
                nme = name.split('_')[0]
                # ac dataset
                replace = (("_rgb_anon", "_gt_labelTrainIds"), ("acdc_trainval", "acdc_gt"), ("rgb_anon", "gt"), ("dannet_pred/" + self.set , self.set + '/' + str(nme)))
                nm = img_file
                for r in replace: 
                    nm = nm.replace(*r)
                label_file = nm  
            else: 
                # nc dataset
                replace = (("dannet_pred", "NightCity-label/label"), ('.png', '_labelIds.png')) 
                nm = img_file
                for r in replace:
                    nm = nm.replace(*r) 
                lst = nm.split('/') 
                lst.insert(-1, 'trainid') 
                label_file = '/'.join(lst)   
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
        transforms_compose_label = transforms.Compose([
                            transforms.Resize((512,1024), interpolation=Image.NEAREST)])  
        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  ## image net mean and std  
        
        try: 
            label = Image.open(datafiles["label"])  
            if self.args.random_bin_augment and (not self.set == 'val'): 
                rand = random.randint(0, 49) #inclusive  
                mask_path = os.path.join(self.args.mask_main_path, self.mask_lst[rand])  
                if 'rgb_anon' in name:
                    # ac gt path will be 
                    gt_path = datafiles["label"].replace('_gt_labelTrainIds','_gt_labelColor')       
                else: 
                    # nc gt path will be 
                    replace = (('/label/', '/color/'), ('_labelIds', '_color'), ('/trainid/', '/'))
                    gt_path = datafiles["label"] 
                    for r in replace:
                        gt_path = gt_path.replace(*r)
                per_gt = perturb_gt_gen(mask_path, gt_path, pred_path = datafiles['img'])      
                image = per_gt 
            elif self.args.random_bin_augment and (self.set == 'val'): 
                if 'rgb_anon' in name:
                    # ac fixed images validation
                    path = os.path.join(self.per_gt_val_path_ac, name)
                    image = Image.open(path).convert('RGB') 
                else:
                    # # nc fixed images validation
                    # path = os.path.join(self.per_gt_val_path_nc, name)
                    # image = Image.open(path).convert('RGB') 
                    pass 
            else: 
                image = Image.open(datafiles["img"]).convert('RGB') 
            
            if self.args.augment: 
                if self.set == 'val':
                    if self.args.norm:
                        transforms_compose_img = transforms.Compose([ 
                                transforms.ToTensor(), 
                                transforms.Normalize(*mean_std)
                            ])
                    else: 
                        transforms_compose_img = transforms.Compose([ 
                            transforms.ToTensor()
                        ]) 
                    image = transforms_compose_img(image)   
                    image = torch.tensor(np.array(image)).float() 
                    label = torch.tensor(np.array(label))
                else:
                    joint_transform_mod = joint_transforms.RandomHorizontallyFlip()  
                    image, label = joint_transform_mod(image, label)  
                    if self.args.norm: 
                        transforms_compose_img = transforms.Compose([
                                # transforms.RandomCrop((1024,1024)), 
                                transforms.Resize((512,1024)), 
                                transforms.ToTensor(),
                                transforms.Normalize(*mean_std), # optional feature to check ok or not
                                # transforms.RandomErasing()        
                            ]) 
                    else: 
                        transforms_compose_img = transforms.Compose([
                            # transforms.RandomCrop((1024,1024)), 
                            transforms.Resize((512,1024)), 
                            transforms.ToTensor(),
                            # transforms.RandomErasing()
                        ]) 
                    image = transforms_compose_img(image)  
                    image = torch.tensor(np.array(image)).float()
                    label = transforms_compose_label(label)  
                    label = torch.tensor(np.array(label))     
            else: 
                if self.set == 'val': 
                    image = torch.tensor(np.array(image)).float() 
                    label = torch.tensor(np.array(label))
                else:
                    joint_transform_mod = joint_transforms.RandomHorizontallyFlip()  
                    image, label = joint_transform_mod(image, label)   
                    transforms_compose_img = transforms.Compose([
                            transforms.RandomCrop((1024,1024))]) 
                    image = transforms_compose_img(image) 
                    image = torch.tensor(np.array(image)).float() 
                    label = transforms_compose_label(label) 
                    label = torch.tensor(np.array(label))
        
            img = np.array(image.permute(1,2,0) * 255) 
            img = img.astype(np.uint8)
            im = Image.fromarray(img) 
            im.save('try.png') 
            lb = Image.fromarray(np.array(label)) 
            lb.save('try_label.png') 
        
        except Exception as e: 
            print(e) 
        
        return image, label, name
       
def init_model(args): 
    if args.small_model: 
        model = network.unet.Unet(input_channels=args.num_ip_channels, num_classes=args.num_classes, num_filters=[32,64,128,192], initializers={'w':'he_normal', 'b':'normal'}, dropout=args.dropout)  
    else: 
        # Deeplabv3+ model for prior net
        model = network.modeling.__dict__[args.model](num_classes=args.num_classes, output_stride=args.output_stride, num_ip_channels = args.num_ip_channels)   
        network.utils.set_bn_momentum(model.backbone, momentum=0.01)  
        # network.convert_to_separable_conv(model.classifier) ## seperable conv adding as an extra thing over the normal dannnet but not in pretrain so not using currently  
        if args.restore_from != 'None' and args.restore_from_color_mapping == 'None':
            params = torch.load(args.restore_from)['model_state']
            model.load_state_dict(params)
            print('----------Model initialize with weights from-------------: {}'.format(args.restore_from)) 
        elif args.restore_from != 'None' and args.restore_from_color_mapping != 'None':
            params = torch.load(args.restore_from_color_mapping)
            model.load_state_dict(params)
            print('----------Model initialize with weights from color mapping model-------------: {}'.format(args.restore_from_color_mapping))
        # network.convert_to_separable_conv(model.classifier)
        
    if args.multigpu:
        model = nn.DataParallel(model)
    if args.eval_prior:
        model.eval().cuda()
        print('Mode --> Eval')
    else:
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
            
            if self.args.norm or self.args.augment: 
                label_perturb_tensor = image.detach().clone() 
            else: 
                label_perturb_tensor = image.permute(0, 3, 1, 2).detach().clone() 
        
            seg_pred = self.model(label_perturb_tensor.float().cuda()) 
            seg_label = seg_label.long().cuda() # cross entropy   
            
            # cityscapes weight propotion {similar to the one used by DaNNet}
            weights = torch.log(torch.FloatTensor([0.36869696, 0.06084986, 0.22824049, 0.00655399, 0.00877272, 0.01227341,
                                            0.00207795, 0.0055127, 0.15928651, 0.01157818, 0.04018982, 0.01218957,
                                            0.00135122, 0.06994545, 0.00267456, 0.00235192, 0.00232904, 0.00098658,
                                                0.00413907])).cuda() 
            # weights 
            weights = (torch.mean(weights) - weights) / torch.std(weights) * 0.16 + 1.0  
            
            if self.args.ignore_classes:
                # black out few classes 
                loss = CrossEntropy2d(ignore_classes=args.ignore_classes, weights=weights)  
            else:
                loss = nn.CrossEntropyLoss(ignore_index=255, weight=weights) 
                
            seg_loss = loss(seg_pred, seg_label)   
            total_loss += seg_loss.item()
    
        total_loss /= len(iter(testloader))
        print('---------------------')
        print('Validation seg loss: {} at epoch {}'.format(total_loss,epoch))
        return total_loss

class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255, ignore_classes = [], weights = None):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label
        self.ignore_classes = ignore_classes
        self.weights = weights

    def forward(self, predict, target):
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
        
        target_mask = (target >= 0) * (target != self.ignore_label)  
        
        if self.ignore_classes:
            self.ignore_classes = json.loads(self.ignore_classes[0])  # to convert string of list to int of list 
            ## have to ignore classes  
            for ig_cl in self.ignore_classes: 
                target_mask = target_mask * (target!= ig_cl) 
                   
        target = target[target_mask]
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=self.weights, size_average=self.size_average)
        # loss = F.nll_loss(torch.log(predict), target, weight=weight, size_average=self.size_average) ## NLL loss cause the pred is now in softmax form.. 
        return loss

class Trainer(BaseTrainer):
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.da_model, self.lightnet, self.weights = pred(self.args.num_classes, self.args.model_dannet, self.args.restore_from_da, self.args.restore_light_path)  
        self.da_model = self.da_model.eval() 
        self.lightnet = self.lightnet.eval() 
        
    def iter(self, batch):
        
        image , seg_label, name = batch         
        if self.args.norm or self.args.augment: 
            label_perturb_tensor = image.detach().clone() 
        else: 
            label_perturb_tensor = image.permute(0, 3, 1, 2).detach().clone() 
        
        seg_pred = self.model(label_perturb_tensor.float().cuda()) 
        seg_label = seg_label.long().cuda() # cross entropy  
        
        if self.args.weighted_ce and self.args.ignore_classes: 
            # cityscapes weight propotion 
            weights = torch.log(torch.FloatTensor([0.36869696, 0.06084986, 0.22824049, 0.00655399, 0.00877272, 0.01227341,
                                            0.00207795, 0.0055127, 0.15928651, 0.01157818, 0.04018982, 0.01218957,
                                            0.00135122, 0.06994545, 0.00267456, 0.00235192, 0.00232904, 0.00098658,
                                                0.00413907])).cuda() 
            # 
            weights = (torch.mean(weights) - weights) / torch.std(weights) * 0.05 + 1.0
            loss = CrossEntropy2d(ignore_classes=args.ignore_classes, weights=weights) 
            
        
        elif self.args.weighted_ce: 
            ## weighted cross entropy loss         
            if self.args.naive_weighting:   
                weights = {0: 0.30354092597961424, 1: 0.0736604881286621, 2: 0.17741899490356444, 3: 0.01694552421569824, 4: 0.012325115203857422, 5: 0.00943770408630371, 6: 0.0021120643615722655, 7: 0.003739910125732422, 8: 0.11350025177001953, 9: 0.006554098129272461, 10: 0.14367461204528809, 11: 0.0014812946319580078, 12: 0.0003917694091796875, 13: 0.021362504959106444, 14: 0.00021363258361816405, 15: 0.0021049118041992186, 16: 0.0072266006469726566, 17: 0.00041659355163574217, 18: 0.0007854843139648437} 
                weights = torch.FloatTensor([weights[key] for key in weights]).cuda() 
                 
            elif self.args.dannet_acdc_weighting:  
                weights = {0: 0.30354092597961424, 1: 0.0736604881286621, 2: 0.17741899490356444, 3: 0.01694552421569824, 4: 0.012325115203857422, 5: 0.00943770408630371, 6: 0.0021120643615722655, 7: 0.003739910125732422, 8: 0.11350025177001953, 9: 0.006554098129272461, 10: 0.14367461204528809, 11: 0.0014812946319580078, 12: 0.0003917694091796875, 13: 0.021362504959106444, 14: 0.00021363258361816405, 15: 0.0021049118041992186, 16: 0.0072266006469726566, 17: 0.00041659355163574217, 18: 0.0007854843139648437} 
                weights = torch.log(torch.FloatTensor([weights[key] for key in weights])).cuda() 
                weights = (torch.mean(weights) - weights) / torch.std(weights) * 0.05 + 1.0 
            else: 
                # cityscapes weight propotion {similar to the one used by DaNNet}
                weights = torch.log(torch.FloatTensor([0.36869696, 0.06084986, 0.22824049, 0.00655399, 0.00877272, 0.01227341,
                                                0.00207795, 0.0055127, 0.15928651, 0.01157818, 0.04018982, 0.01218957,
                                                0.00135122, 0.06994545, 0.00267456, 0.00235192, 0.00232904, 0.00098658,
                                                    0.00413907])).cuda() 
                # weights 
                weights = (torch.mean(weights) - weights) / torch.std(weights) * 0.05 + 1.0  
            
            loss = nn.CrossEntropyLoss(ignore_index=255, weight=weights) 
        elif self.args.ignore_classes:
            loss = CrossEntropy2d(ignore_classes=args.ignore_classes)  
        else: 
            loss = nn.CrossEntropyLoss(ignore_index=255)
        
        seg_loss = loss(seg_pred, seg_label)        
        self.losses.seg_loss = seg_loss
        loss = seg_loss  
        loss.backward()   

    def train(self):
        writer = SummaryWriter(comment=self.args.save_writer_name)
        
        if self.args.multigpu:       
            # self.optim = optim.SGD(self.model.parameters(),
            #               lr=self.args.learning_rate, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
            self.optim = optim.Adam(self.model.parameters(),
                        lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        else:
            # self.optim = optim.SGD(self.model.parameters(),
            #               lr=self.args.learning_rate, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
            self.optim = optim.Adam(self.model.parameters(),
                        lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        
        self.loader = init_train_data(self.args)  
        
        best_val_epoch_loss = float('inf') 
        train_epoch_loss = 0 
        max_decay_steps = self.args.epochs 
        # max_iter = self.args.epochs * self.args.batch_size 
        print('Lets ride...') 

        for epoch in tqdm(range(self.args.epochs)):
            epoch_infor = ('epoch = {:6d}/{:6d}, exp = {}'.format(epoch, int(self.args.epochs), self.args.save_writer_name))
            print(epoch_infor) 
            self.model = self.model.train()  
            
            if args.scheduler: 
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
                        
        print('best_val_epoch_loss: ', best_val_epoch_loss, 'best_epoch:', best_epoch)
        writer.export_scalars_to_json("./all_scalars.json")
        writer.close()


def main(args): 
    if args.multigpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5" 
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
        
            
            
            
            




            

    
    
    

    
    
    
    
    
        

    
    
    
    
    
