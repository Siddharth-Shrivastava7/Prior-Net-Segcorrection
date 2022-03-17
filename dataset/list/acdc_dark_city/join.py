import os
import numpy as np
from PIL import Image 
import shutil
from tqdm import tqdm

## org

# path = '/home/sidd_s/scratch/data_hpc/data/acdc_gt/gt/night/train'
# path = '/home/sidd_s/scratch/data_hpc/data/cityscapes/black_out_val'
# lst_dirs = os.listdir(path)

# save_path = '/home/sidd_s/scratch/data/acdc_city_dark/val/img'

# for direc in tqdm(lst_dirs):
#     lst_imgs = os.listdir(os.path.join(path, direc))
#     for img in tqdm(lst_imgs):
#         # if img.endswith('_labelIds.png'):
#         shutil.copy(os.path.join(path, direc, img), save_path)
        
# print('yo')

## org

## rename
# path = '/home/sidd_s/scratch/data/acdc_city_dark/train/gt'
# lst = os.listdir(path)

# for img in tqdm(lst):
#     # im = Image.open(os.path.join(path,img))
#     # im.save(os.path.join(path, img.replace('_gt_labelIds.png', '_gtFine_labelIds.png')))
#     shutil.move(os.path.join(path,img),os.path.join(path, img.replace('_gt_labelIds.png', '_gtFine_labelIds.png')))
    
# print('yo')
## remane

## list 
# path = '/home/sidd_s/scratch/data/acdc_city_dark/val/img'

# lst = os.listdir(path)

# with open('val.txt','w') as f:
#     for img in tqdm(lst):
#         f.write(img)
#         f.write('\n')
#     f.close()

# print('yo')

## list

### fake gt creation

# path_gt = '/home/sidd_s/scratch/data_hpc/data/acdc_gt/gt/night/train'
# path_rf = '/home/sidd_s/scratch/data/acdc_gt/rf_fake_dannet_train'
# lst_dirs = os.listdir(path_gt)

# save_path = '/home/sidd_s/scratch/data/acdc_gt/acdc_fake_ce_real_en_train'

# for direc in tqdm(lst_dirs):
#     lst_imgs = os.listdir(os.path.join(path_gt, direc))
#     for img in tqdm(lst_imgs):
#         if img.endswith('_gt_labelTrainIds.png'):
#             img_gt = np.array(Image.open(os.path.join(path_gt, direc, img)))
#             nm = img.replace('_gt_labelTrainIds.png','_gt_labelColor.png')
#             img_rf = np.array(Image.open(os.path.join(path_rf, nm)))
#             real_indices = np.transpose(np.where(img_rf==127))
#             img_gt[real_indices[:,0], real_indices[:,1]] = 100   ### changing the number assigned to real for real fake distinction
#             img_gt = Image.fromarray(img_gt)
#             img_gt.save(os.path.join(save_path, img)) 
#             # print('yyy')
        
# print('yo')

### fake gt creation
### uncertainty gt label_id creation

# src_path = '/home/sidd_s/scratch/data_hpc/data/dark_zurich_val/gt/val/night/GOPR0356'
src_path = '/home/sidd_s/scratch/data_hpc/data/acdc_gt/gt/night/train'

lst_dir = os.listdir(src_path)
# print(lst_dir)
inv_lst = [] 
gt_lst = []

for direc in tqdm(lst_dir):
    img_lst = os.listdir(os.path.join(src_path, direc))
    for img in tqdm(img_lst): 
        if img.endswith('_gt_invGray.png'):
            inv_lst.append(os.path.join(src_path, direc, img))
            gt_lst.append(os.path.join(src_path, direc, img.replace('_gt_invGray.png', '_gt_labelTrainIds.png')))

# print(len(inv_lst)) # 106 
# print(len(gt_lst)) # 106 
# print(inv_lst)

for i in tqdm(range(len(gt_lst))):
    inv = np.array(Image.open(inv_lst[i]))
    gt = np.array(Image.open(gt_lst[i]))
    # uncertain_indices = np.transpose(np.where(inv==255))
    # gt[uncertain_indices[:,0], uncertain_indices[:,1]] = 255
    certain_indices = np.transpose(np.where(inv==0)) 
    gt[certain_indices[:,0], certain_indices[:,1]] = 255 ## focusing on uncertain regions only (data uncertain ones....aleatoric)
    img_gt = Image.fromarray(gt) 
    img_gt.save(os.path.join(src_path, gt_lst[i].replace('_gt_labelTrainIds.png', '_gt_label_validinv_TrainIds.png')))

print('yo')


# print(len(gt_lst))
# print(gt_lst)
# print(len(inv_lst))
# print(inv_lst) 


    