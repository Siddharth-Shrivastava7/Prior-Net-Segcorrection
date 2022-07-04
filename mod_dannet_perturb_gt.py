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


parser = argparse.ArgumentParser() 

parser.add_argument("--gpu_id", type=str, default='6',
                        help="GPU ID") 
available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16]) 
# parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
#                         choices=available_models, help='model name') 
parser.add_argument("--model", type=str, default='unet',
                    help='model name') 
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
parser.add_argument("--num_patch_perturb", type=int, default=100,  # may be req 
                        help="num of patches to be perturbed") 
parser.add_argument("--patch_size", type=int, default=10, # may be req 
                        help="size of square patch to be perturbed")
parser.add_argument("--save_writer_name", '-svnm', type=str, default='1024_augment_bt16',
                        help='saving tensorboard file name')  # may be req  
parser.add_argument("--learning_rate", '-lr', type=float, default=3e-4, # may be req  
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
parser.add_argument("--batch_size", type=int, default=16)  # may be req 
parser.add_argument("--worker", type=int, default=4)   
parser.add_argument("--train_dataset", '-tr_nm', type=str, default='synthetic_manual_train_label',
                        choices = ['acdc_train_label','synthetic_manual_train_label', 'synthetic_acdc_train_label', 'dannet_pred_train'], help='train datset name') 
parser.add_argument("--val_dataset", '-val_nm', type=str, default='synthetic_manual_val_label',
                        choices = ['acdc_val_label', 'synthetic_manual_val_label', 'synthetic_acdc_val_label', 'dannet_pred_val'], help='valditation datset name') 
parser.add_argument("--print_freq", type=int, default=1)  
parser.add_argument("--epochs", type=int, default=1500)   # may be req 
parser.add_argument("--snapshot", type=str, default='../scratch/saved_models/acdc/dannet',
                        help='model saving directory')   
parser.add_argument("--exp", '-e', type=int, default=1, help='which exp to run')                     
parser.add_argument("--synthetic_perturb", type=str, default='synthetic_manual_dannet_20n_100p_1024im', # manual color decided acc to seperablity b/w classes
                        help='evalutating technique')
parser.add_argument("--end_learning_rate", type=float, default=1e-5, # may be req  
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


args = parser.parse_args()

## dataset init 
def init_train_data(args): 
    
    trainloader = data.DataLoader(
            BaseDataSet(args, args.train_data_dir, args.train_data_list, args.train_dataset, args.num_classes, ignore_label=255, set='train'),
            batch_size=args.batch_size, shuffle=True, num_workers=args.worker, pin_memory=True) 
    
    return trainloader 

## dataset init 
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
        if self.dataset == 'acdc_train_label' or self.dataset == 'acdc_val_label':  
            
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
                
        elif 'synthetic' in self.dataset: 
            for name in self.img_ids: 
                img_file = os.path.join(self.root, name.split('night')[0], 'night/synthetic', self.set, self.args.synthetic_perturb, name.split('/')[-1])
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
                 
        elif 'dannet' in self.dataset: 
            for name in self.img_ids: 
                img_file = os.path.join(self.root, name.split('night')[0], 'night/dannet_pred', self.set, name.split('/')[-1])
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
        
        transforms_compose_label = transforms.Compose([transforms.Resize((self.args.img_size,self.args.img_size),interpolation=Image.NEAREST)])     
        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  ## image net mean and std  
        try:  
            if self.dataset == 'acdc_train_label' or self.dataset == 'acdc_val_label': 
                
                transforms_compose_img = transforms.Compose([
                                transforms.Resize((540, 960)),
                                transforms.ToTensor(),
                                transforms.Normalize(*mean_std) 
                            ]) 
                transforms_compose_label_val = transforms.Compose([
                            transforms.Resize((1080,1920), interpolation=Image.NEAREST)]) 
                if self.set=='val':  
                    ## image for transforming  
                    image = Image.open(datafiles["img"]).convert('RGB') 
                    image = transforms_compose_img(image)    
                    image = torch.tensor(np.array(image)).float() 
                    
                    label = Image.open(datafiles["label"]) 
                    label = transforms_compose_label_val(label)   
                    label = torch.tensor(np.array(label))  
                    
                else: 
                    image = Image.open(datafiles["img"]).convert('RGB')  
                    img_trans = transforms_compose_img(image) 
                    image = torch.tensor(np.array(img_trans)).float() 
                    
                    label = Image.open(datafiles["label"]) 
                    label = transforms_compose_label(label)   
                    label = torch.tensor(np.array(label))   
            
            elif ('synthetic' and 'manual') or ('dannet' and 'pred') in self.dataset:  
                image = Image.open(datafiles["img"]).convert('RGB') 
                label = Image.open(datafiles["label"]) 
                
                if self.args.norm: 
                    transforms_compose_img = transforms_compose_img = transforms.Compose([ 
                                transforms.ToTensor(),
                                transforms.Normalize(*mean_std) ## use only in training 
                            ]) 
                    image = transforms_compose_img(image)  
                    if self.set == 'val': 
                        label = torch.tensor(np.array(label))
                    else: 
                        label = transforms_compose_label(label) 
                        label = torch.tensor(np.array(label))   
                        
                elif self.args.augment: 
                    if self.set == 'val':
                        transforms_compose_img = transforms.Compose([ 
                                transforms.ToTensor(),
                                transforms.Normalize(*mean_std)  
                            ])
                        image = transforms_compose_img(image)   
                        image = torch.tensor(np.array(image)).float() 
                        label = torch.tensor(np.array(label))
                    else:
                        # joint_transforms_list = [
                        #     joint_transforms.RandomSizeAndCrop(args.img_size, False, pre_size=None,
                        #             scale_min=0.5, scale_max=1.0, ignore_index=255),
                        #             joint_transforms.Resize(args.img_size),
                        #             joint_transforms.RandomHorizontallyFlip()] 
                        # joint_transform_mod = joint_transforms.Compose(joint_transforms_list) 
                        
                        joint_transform_mod = joint_transforms.RandomHorizontallyFlip()  
                        image, label = joint_transform_mod(image, label)  
                        transforms_compose_img = transforms.Compose([
                                transforms.ColorJitter(brightness=0.5, contrast=0.5),
                                transforms.ToTensor(),
                                transforms.Normalize(*mean_std)   
                            ]) 
                        image = transforms_compose_img(image)   
                        image = torch.tensor(np.array(image)).float() 
                        label = transforms_compose_label(label)  
                        label = torch.tensor(np.array(label))   
                    
                else: 
                    image = torch.tensor(np.array(image)).float() 
                    if self.set == 'val': 
                        label = torch.tensor(np.array(label))
                    else: 
                        label = transforms_compose_label(label) 
                        label = torch.tensor(np.array(label))   
                     
                                     
            elif 'synthetic' and 'acdc' in self.dataset: 
                transforms_compose_img = transforms.Compose([
                        transforms.Resize((self.args.img_size, self.args.img_size)),
                    ]) 
                if self.set == 'val':
                    image = Image.open(datafiles["img"]).convert('RGB')  
                    image = torch.tensor(np.array(image)).float()  
                    label = Image.open(datafiles["label"]) 
                    label = torch.tensor(np.array(label))  
                else:
                    image = Image.open(datafiles["img"]).convert('RGB')   
                    image = transforms_compose_img(image)   
                    image = torch.tensor(np.array(image)).float() 
                    label = Image.open(datafiles["label"]) 
                    label = transforms_compose_label(label) 
                    label = torch.tensor(np.array(label))   
    
        
        except: 
            # print('**************') 
            print(index)
            index = index - 1 if index > 0 else index + 1 
            return self.__getitem__(index) 

        return image, label, name

# ############mean and std of the dataset#####################

# testdataset = BaseDataSet(args, args.train_data_dir, args.train_data_list, args.train_dataset, args.num_classes, ignore_label=255, set='train')
# testloader = data.DataLoader(
#     testdataset,
#     batch_size=len(testdataset), shuffle=True, num_workers=args.worker, pin_memory=True)


# def mean_std(loader):
#     images, _ , _ = next(iter(loader))
#     # shape of images = [b,c,w,h]
    
#     mean, std = images.mean([0,2,3]), images.std([0,2,3]) 
#     return mean, std

# mean, std = mean_std(testloader)
# print("data mean and data std: \n", mean, std) 

# ## image net mean and std  

# mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
# mean_std = torch.tensor(mean_std) 

# diff_mean, diff_std = abs(mean_std[0] - mean) , abs(mean_std[1] - std)
# print("diff in mean and diff in std: \n", diff_mean, diff_std)
# #  tensor([0.1349, 0.0806, 0.0136]) tensor([0.0575, 0.0746, 0.0430]) for val
# #  tensor([0.1125, 0.0749, 0.0312]) tensor([0.0474, 0.0707, 0.0355]) for train 

# breakpoint()

##############mean and std of the dataset######################

       
def init_model(args): 
    if args.small_model: 
        
        pass 
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
            if self.args.val_dataset == 'acdc_val_label':
                ## perturbation 
                with torch.no_grad():
                    r = self.lightnet(image.cuda())  
                    enhancement = image.cuda() + r
                    if self.args.model_dannet == 'RefineNet':
                        output2 = self.da_model(enhancement)
                    else:
                        _, output2 = self.da_model(enhancement) 
                
                output2 = self.interp_whole(output2).cpu().numpy() 
                seg_pred = torch.tensor(output2) 
                
                seg_pred = F.softmax(seg_pred, dim=1) 
                
                ## converting the gt seglabel with train ids into onehot matrix 
                seg_label = [seg_label[bt] for bt in range(seg_label.shape[0])]  
                for label in seg_label:
                    label[label==255] = 19 
                seg_label = torch.stack(seg_label, dim=0)
                seg_pred = torch.argmax(seg_pred, dim=1)  
                
                for _ in range(self.args.num_patch_perturb): # number of patches to be cropped
                    randx = np.random.randint(seg_pred.shape[1]-self.args.patch_size)
                    randy = np.random.randint(seg_pred.shape[2]-self.args.patch_size)    
                    seg_label[:, randx: randx + self.args.patch_size, randy:randy+self.args.patch_size]  \
                                = seg_pred[:, randx: randx + self.args.patch_size, randy:randy+self.args.patch_size]
                
                if self.args.channel19:
                    ## gt perturb one hot 
                    gt_onehot_batch = F.one_hot(seg_label.to(torch.int64), 20) 
                    gt_onehot_batch = gt_onehot_batch.permute(0, 3, 1, 2) 
                    gt_onehot_batch = gt_onehot_batch[:, :19, :, :]  
                    label_perturb_tensor = gt_onehot_batch.detach().clone() # dont backprop on this   
                else: 
                    gt_3ch = [torch.tensor(label_img_to_color(seg_label[bt])) for bt in range(seg_label.shape[0])]   
                    gt_3ch = torch.stack(gt_3ch, dim=0)
                    label_perturb_tensor = gt_3ch.permute(0, 3, 1, 2).detach().clone()  
            
            elif 'synthetic' or 'dannet' in self.args.val_dataset:  
                if self.args.norm or self.args.augment: 
                    label_perturb_tensor = image.detach().clone() 
                else: 
                    label_perturb_tensor = image.permute(0, 3, 1, 2).detach().clone() 
        
            seg_pred = self.model(label_perturb_tensor.float().cuda()) 
            seg_label = seg_label.long().cuda() # cross entropy   
            
            if self.args.ignore_classes:
                # black out few classes 
                loss = CrossEntropy2d(ignore_classes=args.ignore_classes)  
            else:
                loss = nn.CrossEntropyLoss(ignore_index=255) 
                
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
        self.interp = nn.Upsample(size=(self.args.img_size, self.args.img_size), mode='bilinear', align_corners=True)
        self.interp_whole = nn.Upsample(size=(1080, 1920), mode='bilinear', align_corners=True)
        
    def iter(self, batch):
        
        image , seg_label, name = batch         
        
        if self.args.train_dataset == 'acdc_train_label':
            ## perturbation 
            with torch.no_grad():
                r = self.lightnet(image.cuda())  
                enhancement = image.cuda() + r
                if self.args.model_dannet == 'RefineNet':
                    output2 = self.da_model(enhancement)
                else:
                    _, output2 = self.da_model(enhancement)

            output2 = self.interp(output2).cpu().numpy() 
            seg_pred = torch.tensor(output2) 

            # print(pred.shape)  
            seg_pred = F.softmax(seg_pred, dim=1)       

            ## converting the gt seglabel with train ids into onehot matrix 
            seg_label = [seg_label[bt] for bt in range(seg_label.shape[0])]  
            for label in seg_label:
                label[label==255] = 19 
            seg_label = torch.stack(seg_label, dim=0)
            seg_pred = torch.argmax(seg_pred, dim=1) 
            
            for _ in range(self.args.num_patch_perturb): # number of patches to be cropped
                randx = np.random.randint(seg_pred.shape[1]-self.args.patch_size)
                randy = np.random.randint(seg_pred.shape[2]-self.args.patch_size)    
                seg_label[:, randx: randx + self.args.patch_size, randy:randy+self.args.patch_size]  \
                            = seg_pred[:, randx: randx + self.args.patch_size, randy:randy+self.args.patch_size]
            
            if self.args.channel19:
                ## gt perturb one hot 
                gt_onehot_batch = F.one_hot(seg_label.to(torch.int64), 20) 
                gt_onehot_batch = gt_onehot_batch.permute(0, 3, 1, 2) 
                gt_onehot_batch = gt_onehot_batch[:, :19, :,:]  
                label_perturb_tensor = gt_onehot_batch.detach().clone() # dont backprop on this   
            else: 
                gt_3ch = [torch.tensor(label_img_to_color(seg_label[bt])) for bt in range(seg_label.shape[0])]   
                gt_3ch = torch.stack(gt_3ch, dim=0)
                label_perturb_tensor = gt_3ch.permute(0, 3, 1, 2).detach().clone() 
        
        elif 'synthetic' or 'dannet' in self.args.train_dataset:   
            if self.args.norm or self.args.augment: 
                label_perturb_tensor = image.detach().clone() 
            else: 
                label_perturb_tensor = image.permute(0, 3, 1, 2).detach().clone()

        elif 'synthetic' and 'acdc' in self.args.train_dataset: 
            label_perturb_tensor = image.detach().clone()
        
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
                # cityscapes weight propotion 
                weights = torch.log(torch.FloatTensor([0.36869696, 0.06084986, 0.22824049, 0.00655399, 0.00877272, 0.01227341,
                                                0.00207795, 0.0055127, 0.15928651, 0.01157818, 0.04018982, 0.01218957,
                                                0.00135122, 0.06994545, 0.00267456, 0.00235192, 0.00232904, 0.00098658,
                                                    0.00413907])).cuda() 
                # 
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
                early_stop_patience = 0 ## early stopping variable 
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
        os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3,7,0" 
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
        
            
            
            
            




            

    
    
    

    
    
    
    
    
        

    
    
    
    
    