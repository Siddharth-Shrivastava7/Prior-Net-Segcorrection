import numpy as np
from PIL import Image   
import os 
import argparse
from tqdm import tqdm
import torchvision.transforms as transforms  
import torch  
import random

parser = argparse.ArgumentParser() 
parser.add_argument('--num_pix_perturb', type = int, help='number of pixels to be perturbed', default=100) 
parser.add_argument('--num_patch_perturb', type = int, help='number of patches to be perturbed', default=1) 
parser.add_argument('--patch_size', type = int, help='size (1d) of square shape to be perturbed', default=10)
parser.add_argument('--save_path', type = str, help='saving path', default='1000n_20p_dannet_pred')  
parser.add_argument('--task', type = str, help='which task to perform', default='dannet_pred_patch_perturb')  
parser.add_argument('--num_masks', type = int, help='number of perturb mask to be used in an image', default=1000) 
parser.add_argument('--sq_size', type = int, help='square height/width dimension', default=20) 
   
args = parser.parse_args()  

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
        255: [0,  0, 0], 
        } 
    # img = np.array(img.cpu())
    img_height, img_width = img.shape
    img_color = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    for row in range(img_height):
        for col in range(img_width):
            label = img[row][col] 
            img_color[row, col] = np.array(label_to_color[label])  
    return img_color

# # normal perturbation in gt file using random binary mask
# def perturb_gt_gen(mask_path, gt_path, pred_path, perturb_gt_path):
#     mask =  Image.open(mask_path).convert('L') 
#     gt = Image.open(gt_path) 
#     pred = Image.open(pred_path) 
#     gt = np.array(gt)
#     pred = np.array(pred)
#     big_mask = np.zeros((gt.shape[0], gt.shape[1])) 
#     # single mask is being superimposed  
#     mask =  Image.open(mask_path).convert('L') 
#     mask = np.array(mask.resize((20, 20), Image.NEAREST)) # strange clouds of dimensions
#     randx = np.random.randint(gt.shape[0]-20)
#     randy = np.random.randint(gt.shape[1]-20) 
#     big_mask[randx: randx+20, randy: randy+20] = mask   
#     gt[big_mask==1] = pred[big_mask==1] 
#     per_gt = Image.fromarray(gt) 
#     per_gt.save(perturb_gt_path)  

# advanced perturbation  
def grp_perturb_gt_gen(mask_paths, gt_path, pred_path, perturb_gt_path):
    gt = Image.open(gt_path)  
    pred = Image.open(pred_path) 
    gt = np.array(gt)
    pred = np.array(pred)
    big_mask = np.zeros((gt.shape[0], gt.shape[1]))
    for mask_path in mask_paths:
        mask =  Image.open(mask_path).convert('L')
        mask = np.array(mask.resize((args.sq_size, args.sq_size), Image.NEAREST)) # strange clouds of dimensions
        randx = np.random.randint(gt.shape[0]-args.sq_size)
        randy = np.random.randint(gt.shape[1]-args.sq_size) 
        big_mask[randx: randx+args.sq_size, randy: randy+args.sq_size] = mask  # 20 x 20  
    
    gt[big_mask==255] = pred[big_mask==255] 
    per_gt = Image.fromarray(gt)
    per_gt.save(perturb_gt_path)
    

def main():
    
    perturb_main_gt_path = os.path.join('/home/sidd_s/scratch/dataset/acdc_trainval/rgb_anon/night/synthetic/val', args.save_path)  
    if not os.path.exists(perturb_main_gt_path):
        os.makedirs(perturb_main_gt_path)
    
    transforms_compose_label_val = transforms.Compose([
                        transforms.Resize((1080,1920), interpolation=Image.NEAREST)])
    
    if args.task == 'dannet_pred_patch_perturb':  
        # whole folder perturbtation
        ## mask paths  
        mask_main_path = '/home/sidd_s/scratch/dataset/random_bin_masks/'
        mask_lst = os.listdir(mask_main_path)  
        
        # list of corresponding gts and preds of dannet (if req at any task)
        root_path = '/home/sidd_s/scratch/dataset/dannet_acdc_pred'
        pred_path_lst = [] 
        gt_path_lst = [] 
        for root, dirs, files in os.walk(root_path, topdown=True):
            if ('images/val') in root: 
                for file in sorted(files): 
                    pred_path_lst.append(os.path.join(root, file)) 
                    gt_root = root.replace('images', 'labels') 
                    gt_file = file.replace('_rgb_anon', '_gt_labelColor') 
                    gt_path_lst.append(os.path.join(gt_root, gt_file))      
        # # normal   
        # for ind in tqdm(range(len(gt_path_lst))):   
        #     mask_ind = ind % len(mask_lst) 
        #     name = pred_path_lst[ind].split('/')[-1]    
        #     mask_paths = []
        #     for i in range(args.num_masks):
        #         rand = random.randint(0, 19) #inclusive  
        #         mask_paths.append(os.path.join(mask_main_path, mask_lst[rand])) 
        #     mask_path = os.path.join(mask_main_path, mask_lst[mask_ind])  
        #     gt_path = gt_path_lst[ind] 
        #     pred_path = pred_path_lst[ind]
        #     perturb_gt_path = os.path.join(perturb_main_gt_path, name.replace('_rgb_anon.png','_gt_labelColor.png')) 
        #     grp_perturb_gt_gen(mask_path, gt_path, pred_path, perturb_gt_path)  
        
        # as per small perturbations 
        for ind in tqdm(range(len(gt_path_lst))): 
            name = pred_path_lst[ind].split('/')[-1]  
            name = name.replace('_rgb_anon.png','_gt_labelColor.png')
            mask_paths = []
            for i in range(args.num_masks):
                rand = random.randint(0, 19) #inclusive  
                mask_paths.append(os.path.join(mask_main_path, mask_lst[rand])) 
            gt_path = gt_path_lst[ind] 
            pred_path = pred_path_lst[ind]
            perturb_gt_path = os.path.join(perturb_main_gt_path, name) 
            grp_perturb_gt_gen(mask_paths, gt_path, pred_path, perturb_gt_path) 
        
        
    else:
        root_path = '/home/sidd_s/scratch/dataset/acdc_gt/gt/night/val/' 
        gt_path_lst = []    
        for root, _, files in os.walk(root_path):  
            for file in sorted(files): 
                if file.endswith('_gt_labelTrainIds.png'):
                    gt_path_lst.append(os.path.join(root_path, file.split('_')[0], file)) 
        
        for file in tqdm(gt_path_lst):
            name = file.split('/')[-1]
            label = Image.open(file) 
            label = transforms_compose_label_val(label)  
            label = torch.tensor(np.array(label)) 
            label_perturb = np.array(label)      
    
            if args.task == 'pixels_perturb':  
                for _ in range(args.num_pix_perturb): 
                    randy, randx = np.random.randint(label_perturb.shape[0]), np.random.randint(label_perturb.shape[1])   
                    actual_label = label_perturb[randy, randx]  
                    while actual_label!=255:
                        perturb_label = np.random.randint(19)  
                        if actual_label!= perturb_label: break  
                    if actual_label == 255: 
                        perturb_label = actual_label
                    label_perturb[randy,randx] = perturb_label 
                    
                im = label_img_to_color(label_perturb) 
                name = name.replace('_gt_labelTrainIds', '_gt_labelColor') 
                im = Image.fromarray(im) 
                im.save(os.path.join(perturb_main_gt_path, name)) 
            
            elif args.task == 'patch_perturb':
                for _ in range(args.num_patch_perturb):  
                    randy, randx = np.random.randint(label_perturb.shape[0] - args.patch_size), np.random.randint(label_perturb.shape[1] - args.patch_size)    

                    lp_patch = label[randy:randy+ args.patch_size, randx:randx+ args.patch_size]  
                    
                    for y in range(lp_patch.shape[0]): 
                        for x in range(lp_patch.shape[1]):
                            actual_label = lp_patch[y,x] 
                            while actual_label!=255:
                                perturb_label = np.random.randint(19)  
                                if actual_label!= perturb_label: break  
                            if actual_label == 255: 
                                perturb_label = 255 
                            lp_patch[y,x] = perturb_label
                    
                    label_perturb[randy:randy+ args.patch_size, randx:randx+ args.patch_size] = lp_patch 
                    
                im = label_img_to_color(label_perturb) 
                name = name.replace('_gt_labelTrainIds', '_gt_labelColor') 
                im = Image.fromarray(im)
                im.save(os.path.join(perturb_main_gt_path, name))  
            
    print('finished it')
            
if __name__ == '__main__': 
    main() 

