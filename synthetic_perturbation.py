import numpy as np
from PIL import Image   
import random 
import os 
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--set', type = str, help='deciding which set train or val is to be used', default='val') 
parser.add_argument('--data', type = str, help='deciding ac or nc dataset', default='ac')  
parser.add_argument('--num_masks', type = int, help='number of perturb mask to be used in an image', default=10) 
parser.add_argument('--save_path', type = str, help='saving path', default='random_small_shapes_overlap')  

args = parser.parse_args() 

# normal
# def perturb_gt_gen(mask_path, gt_path, pred_path, perturb_gt_path):
#     mask =  Image.open(mask_path).convert('L')
#     gt = Image.open(gt_path) 
#     pred = Image.open(pred_path) 
#     gt = gt.resize((1024, 512), Image.NEAREST) # for making sure nc dataset consistency shape{512,1024}
#     if args.set == 'train':
#         mask = np.array(mask)
#         mask[mask==255] = 1
#         gt = np.array(gt.resize((mask.shape[0],mask.shape[1]),Image.ANTIALIAS))
#         pred = np.array(pred.resize((mask.shape[0],mask.shape[1]),Image.ANTIALIAS))
#     else:
#         gt = np.array(gt)
#         pred = np.array(pred)
#         mask = np.array(mask.resize((gt.shape[1], gt.shape[0]), Image.NEAREST)) # strange clouds of dimensions
#         mask[mask==255] = 1 
    
#     gt[mask==1] = pred[mask==1] 
#     per_gt = Image.fromarray(gt)
#     per_gt.save(perturb_gt_path)   
 
## to change with small perturbations at a time 
def perturb_gt_gen(mask_paths, gt_path, pred_path, perturb_gt_path, big_mask_path):
    gt = Image.open(gt_path)  
    pred = Image.open(pred_path) 
    gt = np.array(gt)
    pred = np.array(pred)
    big_mask = np.zeros((gt.shape[0], gt.shape[1]))
    for mask_path in mask_paths:
        mask =  Image.open(mask_path).convert('L')
        mask = np.array(mask.resize((10, 10), Image.NEAREST)) # strange clouds of dimensions
        randx = np.random.randint(gt.shape[0]-10)
        randy = np.random.randint(gt.shape[1]-10) 
        big_mask[randx: randx+10, randy: randy+10] = mask  # 10 x 10  
    
    gt[big_mask==255] = pred[big_mask==255] 
    per_gt = Image.fromarray(gt)
    per_gt.save(perturb_gt_path)
    
    # # saving big mask 
    big_mask = Image.fromarray(big_mask).convert('L')  
    big_mask.save(big_mask_path) 

 
# whole folder perturbtation
## mask paths 
mask_main_path = '/home/sidd_s/scratch/dataset/random_bin_masks/'
mask_lst = os.listdir(mask_main_path) 

if not os.path.exists(os.path.join('/home/sidd_s/scratch/dataset/acdc_trainval/rgb_anon/night/synthetic/val/random_binary_masks')):
    os.makedirs('/home/sidd_s/scratch/dataset/acdc_trainval/rgb_anon/night/synthetic/val/random_binary_masks')
    
big_mask_main_path = '/home/sidd_s/scratch/dataset/acdc_trainval/rgb_anon/night/synthetic/val/random_binary_masks'

if args.data == 'ac':
    perturb_main_gt_path = os.path.join('/home/sidd_s/scratch/dataset/acdc_trainval/rgb_anon/night/synthetic', args.set, args.save_path)  
    if not os.path.exists(perturb_main_gt_path):
        os.makedirs(perturb_main_gt_path)
        
    ## for each corresponding prediction image and gt image perturbed with any random binary mask (1 to 50) 
    root_path = '/home/sidd_s/scratch/dataset/dannet_acdc_pred'
    pred_path_lst = [] 
    gt_path_lst = []
    # list of corresponding gts and preds  
    for root, dirs, files in os.walk(root_path, topdown=True):
        if ('images' + '/' + args.set) in root: 
            for file in sorted(files): 
                pred_path_lst.append(os.path.join(root, file)) 
                gt_root = root.replace('images', 'labels') 
                gt_file = file.replace('_rgb_anon', '_gt_labelColor') 
                gt_path_lst.append(os.path.join(gt_root, gt_file)) 
else:
    # nc (only on val) 
    perturb_main_gt_path = os.path.join('/home/sidd_s/scratch/dataset/night_city/synthetic_perturb_dannet_random_size_shape_color_val') 
    pred_path = '/home/sidd_s/scratch/dataset/night_city/dannet_pred/val' #TODO
    gt_path = '/home/sidd_s/scratch/dataset/night_city/NightCity-label/color/val'
    pred_path_lst = [] 
    gt_path_lst = [] 
    for name in sorted(os.listdir(gt_path)):
        gt_path_lst.append(os.path.join(gt_path, name)) 
        pred_path_lst.append(os.path.join(pred_path, name.replace('_color',''))) 

# normal   
# for ind in tqdm(range(len(gt_path_lst))):  
#     if args.set == 'train':
#         mask_ind = random.randint(0, 49) #inclusive 
#     else: 
#         mask_ind = ind % len(mask_lst) 
#     name = pred_path_lst[ind].split('/')[-1]  
#     mask_path = os.path.join(mask_main_path, mask_lst[mask_ind])  
#     gt_path = gt_path_lst[ind] 
#     pred_path = pred_path_lst[ind]
#     perturb_gt_path = os.path.join(perturb_main_gt_path, name) 
#     perturb_gt_gen(mask_path, gt_path, pred_path, perturb_gt_path) 

# as per small perturbations 
for ind in tqdm(range(len(gt_path_lst))):
    if args.set == 'train':
        mask_ind = random.randint(0, 49) #inclusive 
    else: 
        mask_ind = ind % len(mask_lst) 
    name = pred_path_lst[ind].split('/')[-1]  
    mask_paths = []
    for i in range(args.num_masks):
        rand = random.randint(0, 19) #inclusive  
        mask_paths.append(os.path.join(mask_main_path, mask_lst[rand])) 
    gt_path = gt_path_lst[ind] 
    pred_path = pred_path_lst[ind]
    perturb_gt_path = os.path.join(perturb_main_gt_path, name) 
    # big_mask path 
    big_mask_path = os.path.join(big_mask_main_path, name)
    perturb_gt_gen(mask_paths, gt_path, pred_path, perturb_gt_path, big_mask_path) 

    
print('finished it')
     
    
    
    
    














