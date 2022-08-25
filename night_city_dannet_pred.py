import torch  
import torch.multiprocessing as mp
from torch.utils import data 
import torchvision.transforms as transforms  
import os 
import torch.backends.cudnn as cudnn 
import numpy as np  
import random  
import torch.nn as nn 
from tqdm import tqdm 
import torch.nn.functional as F
from dataset.network.dannet_pred import pred    
from utils.optimize import * 
from PIL import Image  
import argparse 
from torchsummary import summary 
import glob 


parser = argparse.ArgumentParser() 

parser.add_argument("--gpu_id", type=str, default='3',
                        help="GPU ID") 
parser.add_argument("--model_dannet", type=str, default='PSPNet',
                        help='seg model of dannet') 
parser.add_argument("--num_classes", type=int, default=19,
                        help="num classes (default: 19)") 
parser.add_argument("--restore_from_da", type=str, default='/home/sidd_s/scratch/saved_models/DANNet/dannet_psp.pth',
                        help='Dannet pre-trained model path') 
parser.add_argument("--restore_light_path", type=str, default='/home/sidd_s/scratch/saved_models/DANNet/dannet_psp_light.pth',
                        help='Dannet pre-trained light net model path') 
parser.add_argument("--data_dir", type=str, default='/home/sidd_s/scratch/dataset/night_city/NightCity-images/images/',
                        help='root nc data directory') 
# parser.add_argument("--input_size", '-ip_sz', type=int, nargs="+", default=[1920, 1080],
#                         help='actual input size')  
parser.add_argument("--worker", type=int, default=4)   
parser.add_argument("--set", type=str, default='val')       
args = parser.parse_args()

def init_data(args): 
    loader = data.DataLoader(
            Night_city(args, args.data_dir, set=args.set),
            batch_size=1, shuffle=True, num_workers=args.worker, pin_memory=True)  
    return loader
    
class Night_city(data.Dataset):
    def __init__(self, args, root, set):
        super().__init__()
        
        self.root = root
        self.args = args
        self.set = set 
        self.files = [] 
        self.search_img_nc = os.path.join(self.root, self.set, '*.png') 
        self.img_ids = glob.glob(self.search_img_nc) 
        self.img_ids.sort() 
        
        for img_file in self.img_ids:
            # nc dataset
            replace = (("NightCity-images", "NightCity-label"), ('images', 'label'), ('.png', '_labelIds.png')) 
            nm = img_file
            for r in replace:
                nm = nm.replace(*r) 
            lst = nm.split('/') 
            lst.insert(-1, 'trainid') 
            label_file = '/'.join(lst) 
            name = img_file.split('/')[-1]  
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
        # same dimension keeping as it in training and validation, as to how they are!
        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  ## image net mean and std 
        try: 
            label = Image.open(datafiles["label"])  
            image = Image.open(datafiles['img']).convert('RGB') 
            
            if self.set=='val':  
                ## image for transforming  
                transforms_compose_img = transforms.Compose([
                    transforms.Resize((540, 960)),
                    transforms.ToTensor(),
                    transforms.Normalize(*mean_std), ## use only in training 
                ]) 
                image = transforms_compose_img(image)    
                image = torch.tensor(np.array(image)).float()
                            
            else: 
                transforms_compose_img = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*mean_std)]) 
                img_trans = transforms_compose_img(image) 
                image = torch.tensor(np.array(img_trans)).float() 
            
            label = torch.tensor(np.array(label))  
            
        except: 
            print('**********') 
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
        if not os.path.exists(os.path.join('/home/sidd_s/scratch/dataset/night_city/dannet_pred', self.args.set)):
            os.makedirs(os.path.join('/home/sidd_s/scratch/dataset/night_city/dannet_pred', self.args.set))
        
        for _, batch in tqdm(enumerate(loader)):
            image, seg_label, name = batch
            name = name[0] # lst to string

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
            
            ## dannet predictions 
            dannet_pred = torch.tensor(label_img_to_color(seg_pred[0], synthetic=False)) 
            dannet_pred = np.array(dannet_pred)
                
            ## saving the dannet seg pred
            dannet_pred_img = Image.fromarray(dannet_pred) 
            path_img = os.path.join('/home/sidd_s/scratch/dataset/night_city/dannet_pred', self.args.set, name)
            dannet_pred_img.save(path_img)

class predictor(BaseTrainer):
    def __init__(self, args):
        self.args = args
        self.da_model, self.lightnet, self.weights = pred(self.args.num_classes, self.args.model_dannet, self.args.restore_from_da, self.args.restore_light_path)  
        self.da_model = self.da_model.eval() 
        self.lightnet = self.lightnet.eval() 
        self.interp_whole = nn.Upsample(size=(512, 1024), mode='bilinear', align_corners=True)

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
