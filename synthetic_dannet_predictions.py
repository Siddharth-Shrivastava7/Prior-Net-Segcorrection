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
import network 
import argparse 
from torchsummary import summary 


parser = argparse.ArgumentParser() 

parser.add_argument("--gpu_id", type=str, default='1',
                        help="GPU ID") 
parser.add_argument("--num_classes", type=int, default=19,
                        help="num classes (default: 19)")  
parser.add_argument("--model_dannet", type=str, default='PSPNet',
                        help='seg model of dannet') 
parser.add_argument("--restore_from_da", type=str, default='/home/sidd_s/scratch/saved_models/DANNet/dannet_psp.pth',
                        help='Dannet pre-trained model path') 
parser.add_argument("--restore_light_path", type=str, default='/home/sidd_s/scratch/saved_models/DANNet/dannet_psp_light.pth',
                        help='Dannet pre-trained light net model path') 
parser.add_argument("--num_patch_perturb", type=int, default=20,  # may be req 
                        help="num of patches to be perturbed") 
parser.add_argument("--patch_size", type=int, default=100, # may be req 
                        help="size of square patch to be perturbed")
parser.add_argument("--weight_decay", '-wd', type=float, default=0.0005,
                    help="weight_decay for sgd optimiser") 
parser.add_argument("--data_dir", type=str, default='/home/sidd_s/scratch/dataset',
                        help='train data directory') 
parser.add_argument("--data_list", type=str, default='./dataset/list/acdc/acdc_valrgb.txt',
                        choices = ['./dataset/list/acdc/acdc_valrgb.txt', './dataset/list/acdc/acdc_trainrgb.txt'],help='list of names of validation files')  
# parser.add_argument("--input_size", '-ip_sz', type=int, nargs="+", default=[1920, 1080],
#                         help='actual input size')  
parser.add_argument("--worker", type=int, default=4)   
parser.add_argument("--set", type=str, default='val')       
parser.add_argument("--synthetic_perturb", type=str, default='synthetic_manual_dannet_20n_100p_1024im',
                        help='evalutating technique')
parser.add_argument("--img_size", type=int, default=1024,  # may be req 
                        help="size of image or label, which is to be trained") 
parser.add_argument("--collague", action='store_true', default=False,
                        help="forming collague")
parser.add_argument("--is_synthetic", action='store_true', default=False,
                        help="which color to use for synthetic maps") 
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
                
                transforms_compose_img = transforms.Compose([transforms.Resize((self.args.img_size,self.args.img_size)), transforms.ToTensor(), transforms.Normalize(*mean_std)]) 
                img_trans = transforms_compose_img(image) 
                image = torch.tensor(np.array(img_trans)).float() 
                
                transforms_compose_label = transforms.Compose([transforms.Resize((self.args.img_size,self.args.img_size), interpolation=Image.NEAREST)]) 
                label = Image.open(datafiles["label"]) 
                label = transforms_compose_label(label)   
                label = torch.tensor(np.array(label))    
                    
        except: 
            print('**************') 
            print(index)
            index = index - 1 if index > 0 else index + 1 
            return self.__getitem__(index) 

        return image, label, name
                
                                
def label_img_to_color(img, synthetic = False): 
    manual_dd = {
    0: [77, 77, 77], 1: [100, 110, 120], 2: [150,100,175], 3: [102, 102, 156], 
    4: [100, 51, 43], 5: [125, 175, 175], 6: [250, 170, 30], 7: [220, 220, 0], 
    8: [107, 142,  35], 9: [70, 74,  20], 10: [50, 100, 200], 11: [190, 153, 153], 12: [140,100, 100], 13: [255,255,255], 14: [15, 25,  80], 15: [50, 50,  190], 16: [10, 80, 90], 17: [180, 60,  50], 18: [120, 50,  60], 19: [0,0,0]
    }
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
            if synthetic:
                img_color[row, col] = np.array(manual_dd[label])  
            else:
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
    
    def validate(self): 
        loader = init_data(self.args)   
        if not os.path.exists(os.path.join(self.args.data_dir, 'acdc_trainval/rgb_anon/night','synthetic', self.args.set, 'synthetic_manual_dannet_collague', args.synthetic_perturb)):
            os.makedirs(os.path.join(self.args.data_dir, 'acdc_trainval/rgb_anon/night','synthetic', self.args.set, 'synthetic_manual_dannet_collague', args.synthetic_perturb))
        if not os.path.exists(os.path.join(self.args.data_dir, 'acdc_trainval/rgb_anon/night','synthetic', self.args.set, args.synthetic_perturb)):
            os.makedirs(os.path.join(self.args.data_dir, 'acdc_trainval/rgb_anon/night','synthetic', self.args.set, args.synthetic_perturb))
        
        for _, batch in tqdm(enumerate(loader)):
            image, seg_label, name = batch
            name = name[0].split('/')[-1] 

            ## perturbation 
            with torch.no_grad():
                r = self.lightnet(image.cuda())  
                enhancement = image.cuda() + r
                if self.args.model_dannet == 'RefineNet':
                    output2 = self.da_model(enhancement)
                else:
                    _, output2 = self.da_model(enhancement) 

            if self.args.set == 'val': 
                ## TODO this after loading  ## have to work on this since it requires 19 channels thus, 19 channel output should also have to be there if we want to put this on the eval script directly!
                weights = torch.log(torch.FloatTensor(
                    [0.36869696, 0.06084986, 0.22824049, 0.00655399, 0.00877272, 0.01227341, 0.00207795, 0.0055127, 0.15928651,
                    0.01157818, 0.04018982, 0.01218957, 0.00135122, 0.06994545, 0.00267456, 0.00235192, 0.00232904, 0.00098658,
                    0.00413907])).cuda()
                weights = (torch.mean(weights) - weights) / torch.std(weights) * 0.16 + 1.0
                weights_prob = weights.expand(output2.size()[0], output2.size()[3], output2.size()[2], 19)
                weights_prob = weights_prob.transpose(1, 3)
                output2 = output2 * weights_prob 
                output2 = self.interp_whole(output2).cpu().numpy() 
            else:
                ## in training have to apply weighted cross entropy loss
                ## so passing the input as it is 
                output2 = self.interp(output2).cpu().numpy()

            seg_pred = torch.tensor(output2) 
            seg_pred = F.softmax(seg_pred, dim=1) 

            ## converting the gt seglabel with train ids into onehot matrix 
            seg_label = [seg_label[bt] for bt in range(seg_label.shape[0])]  
            for label in seg_label:
                label[label==255] = 19 
            seg_label = torch.stack(seg_label, dim=0)
            seg_pred = torch.argmax(seg_pred, dim=1)  
            
            seg_label_perturb = seg_label.clone() 
            
            for _ in range(self.args.num_patch_perturb): # number of patches to be cropped
                randx = np.random.randint(seg_pred.shape[1]-self.args.patch_size)
                randy = np.random.randint(seg_pred.shape[2]-self.args.patch_size)    
                seg_label_perturb[:, randx: randx + self.args.patch_size, randy:randy+self.args.patch_size]  \
                            = seg_pred[:, randx: randx + self.args.patch_size, randy:randy+self.args.patch_size]
            
            ## perturbed seg label synthetic
            gt_3ch_perturb_synthetic = torch.tensor(label_img_to_color(seg_label_perturb[0],synthetic=args.is_synthetic))  
            gt_3ch_perturb_synthetic = np.array(gt_3ch_perturb_synthetic) 
            
            ## saving collague and perturbed seg pred which is to be used  
            if args.collague:
                ## perturbed seg label
                gt_3ch_perturb = torch.tensor(label_img_to_color(seg_label_perturb[0],synthetic=args.is_synthetic))  
                gt_3ch_perturb = np.array(gt_3ch_perturb) 
                # path = os.path.join(self.args.data_dir, 'acdc_trainval/rgb_anon/night','synthetic', self.args.set, 'synthetic_manual_dannet_collague', name)
                # gt_3ch.save(path)
                ## original seg label
                gt_3ch = torch.tensor(label_img_to_color(seg_label[0], synthetic=args.is_synthetic))  
                gt_3ch = np.array(gt_3ch) 
                ## dannet predictions 
                dannet_pred = torch.tensor(label_img_to_color(seg_pred[0], synthetic=args.is_synthetic)) 
                dannet_pred = np.array(dannet_pred)
                imgg = np.hstack((gt_3ch, gt_3ch_perturb)) 
                imgg2 = np.hstack((dannet_pred, gt_3ch_perturb_synthetic))
                imgg3 = np.vstack((imgg, imgg2)) 
                imgg = Image.fromarray(imgg3) 
                path = os.path.join(self.args.data_dir, 'acdc_trainval/rgb_anon/night', 'synthetic', self.args.set, 'synthetic_manual_dannet_collague', args.synthetic_perturb, name)
                imgg.save(path)  
            
            ## saving the synthetic seg pred
            
            gt_3ch_perturb_synthetic_img = Image.fromarray(gt_3ch_perturb_synthetic) 
            path_img = os.path.join(self.args.data_dir, 'acdc_trainval/rgb_anon/night', 'synthetic', self.args.set, args.synthetic_perturb, name)
            gt_3ch_perturb_synthetic_img.save(path_img)

class predictor(BaseTrainer):
    def __init__(self, args):
        self.args = args
        self.da_model, self.lightnet, self.weights = pred(self.args.num_classes, self.args.model_dannet, self.args.restore_from_da, self.args.restore_light_path)  
        self.da_model = self.da_model.eval() 
        self.lightnet = self.lightnet.eval() 
        self.interp_whole = nn.Upsample(size=(1080, 1920), mode='bilinear', align_corners=True)
        self.interp = nn.Upsample(size=(self.args.img_size, self.args.img_size), mode='bilinear', align_corners=True)

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
