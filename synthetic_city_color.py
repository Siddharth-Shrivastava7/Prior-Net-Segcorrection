from ast import arg
import numpy as np
from PIL import Image
import os
from tqdm import tqdm 
from collections import defaultdict
import json
import argparse

parser = argparse.ArgumentParser()  
parser.add_argument("--method_color", '-m_c', type=str, default='manual_color',
                        choices = ['centroid_median', 'centroid_mean','manual_color'], help='to get color for synthetic images') 
parser.add_argument("--set", type=str, default='train',
                        help='training or validation') 
parser.add_argument("--dataset", type=str, default='acdc',
                        help='which dataset to use acdc, cityscapes') 
args = parser.parse_args() 


if args.dataset == 'cityscapes':
    # cityscapes images load 
    list_path = '/home/sidd_s/Prior-Net-Segcorrection/dataset/list/cityscapes/' + args.set + '.txt'
    img_ids = []
    with open(list_path) as f: 
        for item in f.readlines(): 
            fields = item.strip().split('\t')[0]
            if ' ' in fields:
                fields = fields.split(' ')[0]
            img_ids.append(fields)  
    image_names = [] 
    label_names = []  
    root = '/home/sidd_s/scratch/dataset/cityscapes'  

elif args.dataset == 'acdc':
    #  images load 
    list_path = '/home/sidd_s/Prior-Net-Segcorrection/dataset/list/acdc/' + 'acdc_' + args.set  + 'rgb.txt'
    img_ids = []
    with open(list_path) as f: 
        for item in f.readlines(): 
            fields = item.strip().split('\t')[0]
            if ' ' in fields:
                fields = fields.split(' ')[0]
            img_ids.append(fields)  
    image_names = [] 
    label_names = []  
    root = '/home/sidd_s/scratch/dataset/' 


# Mapping of ignore categories and valid ones (numbered from 1-19) # cityscapes 
mapping_20 = { 
    0: 255,
    1: 255,
    2: 255,
    3: 255,
    4: 255,
    5: 255,
    6: 255,
    7: 0,
    8: 1,
    9: 255,
    10: 255,
    11: 2,
    12: 3,
    13: 4,
    14: 255,
    15: 255,
    16: 255,
    17: 5,
    18: 255,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    29: 255,
    30: 255,
    31: 16,
    32: 17,
    33: 18,
    -1: 255
}
def encode_labels(mask):
    label_mask = np.zeros_like(mask)
    for k in mapping_20:
        label_mask[mask == k] = mapping_20[k]
    return label_mask

##########################################################################################
# mean color centroid over all the images
dd = defaultdict(list) 
def color_centroid(img, lb):
    for cls in  np.unique(lb):
        if cls == 255: 
            continue
        else:
            # running mean 
            dd[cls].append(np.mean(img[lb==cls], axis=0))  
    return dd
## synthetic color:: mean of mean color centroid over all the images
syn_dd = defaultdict()
def synthetic_color(dd):
    for cls in range(len(dd)):
        syn_dd[cls] = np.round(np.mean(np.array(dd[cls]), axis=0)).astype(np.int64) # converting the rounding result from float64 to int 64 result # for using these values as the colors in the images
    return syn_dd
def synthetic_images_gen(img, lb, syn_dd):
    syn_img = np.zeros_like(img)
    for cls in range(len(syn_dd)):
        syn_img[lb==cls] = syn_dd[cls]  # how to assign the value of the new color to a specific location in the syn image corresponding to that of original image 
    return syn_img

##########################################################################################
# median
def color_centroid_median(img, lb):
    for cls in  np.unique(lb):
        if cls == 255: 
            continue
        else:
            # running mean 
            dd[cls].append(np.median(img[lb==cls], axis=0))  
    return dd
syn_dd_median = defaultdict()
def synthetic_color_median(dd):
    for cls in range(len(dd)):
        syn_dd_median[cls] = np.round(np.median(np.array(dd[cls]), axis=0)).astype(np.int64) # converting the rounding result from float64 to int 64 result # for using these values as the colors in the images
    return syn_dd_median
def synthetic_images_gen_median(img, lb, syn_dd_median): 
    syn_img = np.zeros_like(img) 
    for cls in range(len(syn_dd_median)):
        syn_img[lb==cls] = syn_dd_median[cls]
    return syn_img

##########################################################################################
# manual color assign: 
manual_dd = defaultdict()  
#   (16)
manual_dd = {
    0: np.array([77, 77, 77]), 1: np.array([100, 110, 120]), 2: np.array([150,100,175]), 3: np.array([102, 102, 156]), 
    4: np.array([100, 51, 43]), 5: np.array([125, 175, 175]), 6: np.array([250, 170, 30]), 7:np.array([220, 220,   0]), 
    8: np.array([107, 142,  35]), 9: np.array([70, 74,  20]), 10: np.array([50, 100, 200]), 11: np.array([190, 153, 153]), 12: np.array([140,100,100]), 13: np.array([255,255,255]), 14: np.array([15, 25,  80]), 15: np.array([50, 50,  190]), 16: np.array([10, 80, 90]), 17: np.array([180, 60,  50]), 18: np.array([120, 50,  60]) 
}
def synthetic_images_gen_manual(img, lb, syn_dd_manual): 
    syn_img = np.zeros_like(img) 
    for cls in range(len(syn_dd_manual)):
        syn_img[lb==cls] = syn_dd_manual[cls]
    return syn_img


if args.dataset == 'cityscapes':
    # collecting the image and label names of the files 
    for img_id in sorted(img_ids): 
        image_names.append(os.path.join(root,'leftImg8bit', args.set, img_id)) 
        label_names.append(os.path.join(root, 'gtFine', args.set, img_id.replace('leftImg8bit', 'gtFine_labelIds')))  
    # create means of class color over all the images in the dataset
    for id in tqdm(range(len(image_names))): 
        img = np.array(Image.open(image_names[id])) 
        lb =  np.array(Image.open(label_names[id]))  
        lb = encode_labels(lb)  # converting the label id into train label id 
        if args.method_color == 'centroid_mean':
            dd = color_centroid(img, lb) 
        elif args.method_color == 'centroid_median':
            dd = color_centroid_median(img, lb) 
        elif args.method_color == 'manual_color': 
            pass

elif args.dataset == 'acdc':
    for name in sorted(img_ids): 
        image_names.append(os.path.join(root, name))
        replace = (("_rgb_anon", "_gt_labelTrainIds"), ("acdc_trainval", "acdc_gt"), ("rgb_anon", "gt"))
        nm = name
        for r in replace: 
            nm = nm.replace(*r) 
        label_names.append(os.path.join(root, nm)) 


if args.method_color == 'centroid_mean':
    # overall mean of the images for their specific class values
    syn_dd = synthetic_color(dd)
elif args.method_color == 'centroid_median':
    syn_dd = synthetic_color_median(dd)
elif args.method_color == 'manual_color': 
    syn_dd = manual_dd


if args.dataset == 'cityscapes':
    if args.method_color == 'centroid_mean':
        # save using those colors 
        with open('synthetic_colors_cityscapes.txt', 'w') as f: 
            f.write(str(syn_dd))       
        # path for saving the colored outputs
        if not os.path.exists(os.path.join(root, 'synthetic', args.set, 'synthetic_cityscapes_mean')):
            os.makedirs(os.path.join(root, 'synthetic', args.set, 'synthetic_cityscapes_mean'))
    elif args.method_color == 'centroid_median':
        with open('median_synthetic_colors_cityscapes.txt', 'w') as f:
            f.write(str(syn_dd))      
        # path for saving the colored outputs
        if not os.path.exists(os.path.join(root, 'synthetic', args.set, 'synthetic_cityscapes_centroid_median')):
            os.makedirs(os.path.join(root, 'synthetic', args.set, 'synthetic_cityscapes_centroid_median')) 
    elif args.method_color == 'manual_color': 
        # path for saving the colored outputs
        with open('manual_synthetic_colors_cityscapes.txt', 'w') as f:
            f.write(str(syn_dd))   
        if not os.path.exists(os.path.join(root, 'synthetic', args.set, 'synthetic_cityscapes_manual_color')):
            os.makedirs(os.path.join(root, 'synthetic', args.set, 'synthetic_cityscapes_manual_color')) 

    ## now coverting the orginal cityscapes labels into synthetic labels (wref to color of cityscapes images)
    for id in tqdm(range(len(label_names))):
        img = np.array(Image.open(image_names[id]))
        lb = np.array(Image.open(label_names[id]))   
        lb = encode_labels(lb) # converting the label id into train label id
        name = image_names[id].split('/')[-1]
        if args.method_color == 'centroid_mean':    
            syn_img = synthetic_images_gen(img, lb, syn_dd) 
            syn_img = Image.fromarray(syn_img)
            syn_img.save(os.path.join(root, 'synthetic', args.set, 'synthetic_cityscapes_mean',  name))
        elif args.method_color == 'centroid_median':
            syn_img = synthetic_images_gen_median(img, lb, syn_dd)
            syn_img = Image.fromarray(syn_img)
            syn_img.save(os.path.join(root, 'synthetic', args.set, 'synthetic_cityscapes_centroid_median',  name))
        elif args.method_color == 'manual_color':
            syn_img = synthetic_images_gen_manual(img, lb, syn_dd)
            syn_img = Image.fromarray(syn_img)
            syn_img.save(os.path.join(root, 'synthetic', args.set, 'synthetic_cityscapes_manual_color',  name))


elif args.dataset == 'acdc':
    if args.method_color == 'manual_color':  
        if not os.path.exists(os.path.join(root, 'acdc_trainval/rgb_anon/night','synthetic', args.set, 'synthetic_cityscapes_manual_color')):
            os.makedirs(os.path.join(root, 'acdc_trainval/rgb_anon/night','synthetic', args.set, 'synthetic_cityscapes_manual_color'))

    ## now coverting the orginal cityscapes labels into synthetic labels (wref to color of cityscapes images)
    for id in tqdm(range(len(label_names))):
        img = np.array(Image.open(image_names[id]))
        lb = np.array(Image.open(label_names[id]))   
        name = image_names[id].split('/')[-1]
        if args.method_color == 'manual_color':
            syn_img = synthetic_images_gen_manual(img, lb, syn_dd)
            syn_img = Image.fromarray(syn_img)
            syn_img.save(os.path.join(root, 'acdc_trainval/rgb_anon/night','synthetic', args.set, 'synthetic_cityscapes_manual_color',  name))
    