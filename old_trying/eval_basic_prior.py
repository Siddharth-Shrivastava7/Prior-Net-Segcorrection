import os, sys
import argparse
import numpy as np
import torch
from model.Unet import * 
from collections import OrderedDict

from torch.utils import data
import torch.nn as nn
import os.path as osp
import yaml
# from utils.logger import Logger  
from dataset.dataset import *
from easydict import EasyDict as edict
from tqdm import tqdm
from PIL import Image
import json
import torchvision
import cv2
from dataset.network.dannet_pred import pred
from utils.calibrationmetrics import *
import matplotlib.pyplot as plt

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--num-classes", type=int, default=19)
    parser.add_argument("--num-channels", type=int, default=19)
    parser.add_argument("--frm", type=str, default='/home/sidd_s/scratch/saved_models/acdc/dannet/train/unet_e2e_acdc_iterate_drp.pth')
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--dataset", type=str, default='darkzurich')
    ## pred label of dannet......for prediction
    # img = torch.argmax(image, dim=1)
    # # print(img.shape)
    # img = label_img_to_color(img.squeeze().cpu().numpy())
    parser.add_argument("--model", default='unet') 
    parser.add_argument("-c", "--calibrationcalc", action='store_true')
    return parser.parse_args()

def print_iou(iou, acc, miou, macc):
    for ind_class in range(iou.shape[0]):
        print('===> {0:2d} : {1:.2%} {2:.2%}'.format(ind_class, iou[ind_class, 0].item(), acc[ind_class, 0].item()))
    print('mIoU: {:.2%} mAcc : {:.2%} '.format(miou, macc))

def compute_iou(model, testloader, args, da_model, lightnet, weights, fake_ce):
    model = model.eval()

    interp = nn.Upsample(size=(1080,1920), mode='bilinear', align_corners=True)   # dark_zurich -> (1080,1920)
    union = torch.zeros(args.num_classes, 1,dtype=torch.float).cuda().float()
    inter = torch.zeros(args.num_classes, 1, dtype=torch.float).cuda().float()
    preds = torch.zeros(args.num_classes, 1, dtype=torch.float).cuda().float()
    # extra 
    gts = torch.zeros(args.num_classes, 1, dtype=torch.float).cuda().float()

    logits_list = []
    labels_list = []
    # 2nd best 
    # totalp = 0
    # Tp = 0   
    with torch.no_grad():
        for index, batch in tqdm(enumerate(testloader)):
            # print('******************') 
            image, label, edge, _, name = batch
            # print(image.shape) 
#            edge = F.interpolate(edge.unsqueeze(0), (512, 1024)).view(1,512,1024)
            # print(name)
            # if name[0].find('dannet_pred')==-1: 
            #     continue
            # print(name)
            # image = F.softmax(image,dim=0)  ## don't use...if you have not used in training time......
            
            interp = nn.Upsample(size=(1080, 1920), mode='bilinear', align_corners=True)
            da_model = da_model.eval()
            lightnet = lightnet.eval()

            with torch.no_grad():
                r = lightnet(image.cuda())
                enhancement = image.cuda() + r
                if model == 'RefineNet':
                    output2 = da_model(enhancement)
                else:
                    _, output2 = da_model(enhancement)
                                    
            weights_prob = weights.expand(output2.size()[0], output2.size()[3], output2.size()[2], 19)
            weights_prob = weights_prob.transpose(1, 3)
            output2 = output2 * weights_prob
            output2 = interp(output2).cpu().numpy()

            image = torch.tensor(output2) 
            # print(image.shape) # torch.Size([1, 19, 1080, 1920]) 

            ### calibration 
            if args.calibrationcalc:
                # print('*****')
                logits_list.append(image) 
                labels_list.append(label)
                logits = torch.cat(logits_list).cuda() 
                labels = torch.cat(labels_list).cuda()   
                # print(logits.shape) 
                # print('*********')
                # print(labels.shape)
                continue

            # print('oooooooo') ## yes this is not coming while calibrationcalc
            
            ## pred label of dannet......for prediction

            # img = torch.argmax(image, dim=1)
            # # print(img.shape)
            # img = label_img_to_color(img.squeeze().cpu().numpy())
            # img = torch.tensor(img)

            ### pred label....

            ### one_hot_conversion.... 
            # img = image
            # nc = img.shape[1]
            # img = torch.argmax(img, dim=1)
            # img = F.one_hot(img, num_classes=nc)
            # # print(img.shape) # torch.Size([12, 512, 512, 19]) 
            # img = img.transpose(3,2).transpose(2,1) 
            # print(img.shape) # torch.Size([12, 19, 512, 512])
            ### one_hot_conversion.... 

            # output =  model(F.softmax(image.cuda(),dim=1))      
            # output = model(img.float().cuda())  ## one hot input  
            # print(image.shape) # torch.Size([1, 19, 1080, 1920])
            output = model(image.cuda()) 
            # output = model(F.softmax(img.float().cuda(), dim = 1)) 
            label = label.cuda()

            # output = output.squeeze()
            # image = image.squeeze() 
            # print(image.shape) # torch.Size([19, 1080, 1920]) 
            # print(output.shape) # torch.Size([19, 1080, 1920])
            B,C,H,W = output.shape 
            # posterior = output * image.cuda()  ## it's logits....which usually have taken to go through cross entropy loss...
            # posterior = F.softmax(posterior, dim=0)
        
            # if args.calibration: 
            #     ## make reliability plot ... 
            #     image_arr = np.array(image)
            #     label_arr = np.array(label)
            #     for i in range(1,11):
            #         indices = image_arr[0<image_arr<0.1] 
                    

            #     ## confidences with 0 to 1 with gap of 0.1 
            #     continue
            
            if fake_ce:
                ## iterate 
                # print(image.shape)  # torch.Size([19, 1080, 1920]) 
                # print(output.shape)  # torch.Size([19, 1080, 1920])  
                posterior = output * image.cuda().detach()  
                # # posterior = image.cuda() 
                # # print(post1.shape)  
                seg_pred2 = model(posterior.cuda())  
                posterior = seg_pred2 * posterior.cuda()   

                ## probability distribution 
                # posterior = posterior.squeeze() 
                # # print(posterior.shape)
                # posterior = F.softmax(posterior, dim=0) 

                # print(posterior.shape)
                # img_prob = F.softmax(image.cuda(), dim=1)  
                # entropy_map = -torch.sum(img_prob*torch.log(img_prob), dim=1) 
                # # print(torch.max(entropy_map))   
                # entropy_map_norm = entropy_map / torch.max(entropy_map) 
                # # print(entropy_map_norm.shape) # torch.Size([12, 512, 512])
                # entropy_map_norm = entropy_map_norm.unsqueeze(dim=1)   

                # out = output.cuda() * entropy_map_norm.detach()  + (1-entropy_map_norm.detach()) * image.cuda().detach()
   
                # # # repeat 
                # seg_pred2 = model(out.cuda())   
                # posterior = seg_pred2.cuda() * entropy_map_norm.detach()  + (1-entropy_map_norm.detach()) * image.cuda().detach()  


                ## probability distribution 
                # posterior = image.cuda()
                posterior = posterior.squeeze() 
                # # print(posterior.shape)
                posterior = F.softmax(posterior, dim=0) 

                # # print('************')
                # # seg_loss = loss(output, label) 
                # ### to add the evaluation.... 
                # # print('will do') 

                # # output = F.softmax(output, dim=0)
                # # image = F.softmax(image, dim= 0)

                # # entropy_prior = -torch.sum(output*torch.log(output), dim=0)
                # entropy_prior = -torch.sum(F.softmax(output, dim=0) * F.log_softmax(output, dim=0), dim=0) 
                # entropy_max = np.max(np.array(entropy_prior.cpu()))
                # entropy_min = np.min(np.array(entropy_prior.cpu()))
                # # print(entropy_max)
                # # print(entropy_min) 
                # entropy_prior = entropy_prior / entropy_max 
                # # entropy_max = np.max(np.array(entropy_prior.cpu()))
                # # entropy_min = np.min(np.array(entropy_prior.cpu()))
                # # print(entropy_max)  # 1 
                # # print(entropy_min)  # 0.89 
                # # entropy_prior = F.softmax(entropy_prior, dim=0)
                # # entropy_max = np.min(np.array(entropy_prior.cpu()))
                # # print(entropy_max) # nan....
                # # threshold entropy_map # not helping much
                # # entropy_prior[entropy_prior>=0.4] = 1
                # # entropy_prior[entropy_prior<0.4] = 0

                # entropy_liki = -torch.sum(F.softmax(image.cuda(), dim=0) * F.log_softmax(image.cuda(), dim=0), dim=0)
                # entropy_max_l = np.max(np.array(entropy_liki.cpu()))
                # entropy_liki = entropy_liki / entropy_max_l

                # name =name[0].split('/')[-1] 
                # en_prior = np.array(255* entropy_prior.cpu()) 
                # en_prior = Image.fromarray(en_prior).convert('L')
                # # print(en_prior.shape)  # (1080, 1920) 
                # # en_prior.save('../scratch/data/e2e_pred/unet_e2e_resize_mod_entropyprior/' + name + '.png') 

                # # print(np.max(np.array(entropy_prior.cpu()))) # nan.....ok..error..
                # # output = F.softmax(output, dim=0) 
                # # image = F.softmax(image, dim=0)
                # # posterior = F.log_softmax(output, dim=0)  + F.log_softmax(image.cuda(),dim=0)

                # lamb = 0.5
                # # posterior = lamb* F.log_softmax(output, dim=0)  +  F.log_softmax( image.cuda(),dim=0)
                # posterior = lamb* entropy_liki * F.log_softmax(output, dim=0)  +   F.log_softmax(image.cuda(),dim=0)  
                # # posterior = lamb* F.log_softmax( (1-entropy_prior) * output, dim=0)  + F.log_softmax(image.cuda(),dim=0)
                # # posterior = lamb* F.log_softmax( entropy_liki * output, dim=0)  + F.log_softmax( entropy_prior * image.cuda(),dim=0) 
                # # posterior = lamb* entropy_liki * F.log_softmax(output, dim=0)  +   F.log_softmax( (1-entropy_liki) * image.cuda(),dim=0)
                # # posterior = lamb* (1-entropy_liki) * F.log_softmax(output, dim=0)  +   F.log_softmax( (1-entropy_prior) * image.cuda(),dim=0)
                # # output = F.log_softmax( (1-entropy_prior) * output, dim=0)

                # # posterior = lamb* F.log_softmax( entropy_liki * output, dim=0)  + F.log_softmax( (1-entropy_liki) * image.cuda(),dim=0)
                # # posterior = lamb* F.log_softmax( entropy_liki * output, dim=0)  + F.log_softmax( image.cuda(),dim=0)
                # # print('*************')
                # # posterior = lamb* F.log_softmax( entropy_liki * output, dim=0)  + F.log_softmax( image.cuda(),dim=0)
                # # posterior = lamb * (1-entropy_prior) * output + image.cuda()
                # # posterior = torch.log((1-entropy_prior) * output) + torch.log(image.cuda())
                # # posterior = (1-entropy_prior) * output * image.cuda()
                # # posterior = output * image.cuda()

            else: 
                # print('label shape:{} output shape:{}'.format(label.shape, output.shape))
                # output = interp(output).squeeze()
                # output = output.squeeze()
                name = name[0].split('/')[-1] 
                # print('****************') 
                # print(output.shape)
                # dannet_pred = output.argmax(dim=0).float()
                # print(name[0].split('/')[-1])

                # output = image.cuda() 
                # output = output.squeeze()
                # C = 2
                # print(output.shape)
                # print('**********')
                # H, W = output.shape
                # C, H, W = output.shape # original
                # print('[*****]')
                # print(C)
                # print(torch.unique(output))
                # print(torch.unique(torch.argmax(output, dim = 0)))

                ### posterior MAP evaluation....
                ##this is working better... not always not working better for fake ce and real en
                # output = F.softmax(output, dim=0)  
                # image = F.softmax(image, dim=0)
                # posterior = output * image.cuda() ## output...if softmax...then seeing...cause image itself is taken proba distribut
                ## this is working better..

                ### another change 
                # image = F.softmax(image,dim=0)  ## we will use it....but first applying it on top of the model...(if not used during training then it doesn't worked)
                # output = F.softmax(output, dim=0) 
                # posterior = output * image.cuda() 
                lamb = 1 ## 0.96 for fake_ce and real_en 
                posterior = lamb * F.log_softmax(output, dim=0)  + F.log_softmax(image.cuda(),dim=0)
                # posterior = lamb* output + image.cuda() 
                # posterior = output * image.cuda()
                # post = posterior.argmax(dim=0).float()  
                # save_pred(post, '../scratch/data/e2e_pred/unet_e2e_resize_mod_post',  name + '.png') # org
                # posterior = lamb * output * image.cuda() 
                ### another change 

                ### another change
                # image = F.log_softmax(image,dim=0)  ## we will use it...but first applying it on top of the model...(if not used during training then it doesn't worked)
                # output = F.log_softmax(output, dim=0) 
                
                # posterior = output + image.cuda() 
                ### another change
                
                # print(label.shape) # ([1, 1080, 1920])
                # print('**************')
                # print(torch.unique(label)) ## wtf....how 0 and 1 only......due to TF functinon of to_tensor...not going to use that now
                # print(image.shape) # torch.Size([19, 1080, 1920])
                # print('********')
                # print(torch.unique(image)) # sort of logits from proba distri...  since then Dannet has done a post processing step where they multiplied with weight matrices in order to get more appro results...but since they are doing the argmax so...i am taking it as it is
                # print(output.shape) # torch.Size([19, 1080, 1920])
                # print('********') 
                # print(torch.unique(output)) # logits 

                ### posterior MAP evaluation....

            #########################################################################original
            Mask = (label.squeeze())<C  # it is ignoring all the labels values equal or greater than 2 #(1080, 1920) 
            pred_e = torch.linspace(0,C-1, steps=C).view(C, 1, 1)  
            pred_e = pred_e.repeat(1, H, W).cuda() 
            # print(pred_e.shape) # torch.Size([19, 1080, 1920])
            # pred = output.argmax(dim=0).float() ## prior 
            pred = posterior.argmax(dim=0).float() ## posterior 
            # pred = image.argmax(dim=0).float().cuda() ## likelihood 
            # print(pred.shape) # torch.Size([1080, 1920]) 
            pred_mask = torch.eq(pred_e, pred).byte()    
            pred_mask = pred_mask*Mask 
            # print(Mask.shape) #torch.Size([1080, 1920])
            # print(pred_mask.shape) #torch.Size([19, 1080, 1920]) 

            label_e = torch.linspace(0,C-1, steps=C).view(C, 1, 1)
            label_e = label_e.repeat(1, H, W).cuda()
            label = label.view(1, H, W)
            label_mask = torch.eq(label_e, label.float()).byte()
            label_mask = label_mask*Mask
            # print(label_mask.shape) # torch.Size([19, 1080, 1920])

            tmp_inter = label_mask+pred_mask
            # print(tmp_inter.shape)  # torch.Size([19, 1080, 1920])
            cu_inter = (tmp_inter==2).view(C, -1).sum(dim=1, keepdim=True).float()
            cu_union = (tmp_inter>0).view(C, -1).sum(dim=1, keepdim=True).float()
            cu_preds = pred_mask.view(C, -1).sum(dim=1, keepdim=True).float()
            # extra
            # cu_gts = label_mask.view(C, -1).sum(dim=1, keepdim=True).float()
            # gts += cu_gts
            union+=cu_union
            inter+=cu_inter
            preds+=cu_preds
            #########################################################################origninal

            ########################################################################with2ndbest
            # Mask = (label.squeeze())<C
            # pred_e = torch.linspace(0,C-1, steps=C).view(C, 1, 1)
            # pred_e = pred_e.repeat(1, H, W).cuda()
            # pred = output.argmax(dim=0).float()  # most confident pred 
            # pred2 = output.topk(2,dim=0)[1][1].float()  # 2nd most confi   
            # pred_mask = torch.eq(pred_e, pred).byte()
            # pred2_mask = torch.eq(pred_e, pred2).byte()
            # pred_mask = pred_mask*Mask
            # pred2_mask = pred2_mask*Mask

            # label_e = torch.linspace(0,C-1, steps=C).view(C, 1, 1)
            # label_e = label_e.repeat(1, H, W).cuda()
            # label = label.view(1, H, W)
            # label_mask = torch.eq(label_e, label.float()).byte()
            # label_mask = label_mask*Mask

            # pred_maskb = torch.logical_or(pred_mask, pred2_mask) # for combining both 1st most and 2 most confident pred       

            # tmp_interb = label_mask+pred_maskb  # for getting top 2 acc 
            # tmp_inter = label_mask + pred_mask  
            # tmp_inter2 = label_mask + pred2_mask   

            # cu_inter = (tmp_inter2==2).view(C, -1).sum(dim=1, keepdim=True).float() # overall including both predictions

            # #####ignore this 
            # cu_union = (tmp_inter2>0).view(C, -1).sum(dim=1, keepdim=True).float() 
            # cu_preds = pred2_mask.view(C, -1).sum(dim=1, keepdim=True).float() 
            # ##### ignore this 

            # Tp += sum((tmp_interb==2).view(C, -1).sum(dim=1, keepdim=True)).float()
            # totalp += H*W      

            # # print(pred_mask.shape)
            # union+=cu_union
            # inter+=cu_inter
            # preds+=cu_preds
            ##########################################################################with2ndbest
        
        ###########original
        iou = inter/union
        acc = inter/preds
        mIoU = iou.mean().item()
        mAcc = acc.mean().item()
        # print('*********')
        # print(gts)
        print_iou(iou, acc, mIoU, mAcc)  ## original
        ##################
        
        # print(logits.shape)  # torch.Size([50, 19, 1080, 1920])
        # print('*********' )
        # print(labels.shape) # torch.Size([50, 1080, 1920])

        if args.calibrationcalc:

            # SCE calibration metric 
            sce_criterion_seg = ClasswiseECELossSeg().cuda()
            sce, class_sce_lst = sce_criterion_seg(logits, labels)   
            print(class_sce_lst)   

            # ACE Calibration metric
            aece_criterion_seg = ClasswiseAdaptiveECELoss().cuda()
            aece, class_aece_lst= aece_criterion_seg(logits, labels)

            save_path = '/home/sidd_s/Prior_Net/class_sce_plots'
            num_classes = len(class_sce_lst)
            # print(len(class_aece_lst[0]))
            for c in tqdm(range(num_classes)): 
                plt.figure()
                acc_lst =  [val[0] for val in class_sce_lst[c]]     
                conf_lst = [val[1] for val in class_sce_lst[c]]
                plt.plot(acc_lst, conf_lst ,'--o', label='class ' + str(c))
                plt.plot([0.0, 1.0], [0.0, 1.0], label = 'perfect')
                plt.title('Classwise AECE for Class ' + str(c)) 
                plt.legend()
                plt.savefig(os.path.join(save_path, 'Class_' + str(c) + '.png')) 
        
            print('yo')             
                
            print(sce.item()) 
            # ACE Calibration metric 
            aece_criterion_seg = ClasswiseAdaptiveECELoss().cuda()
            aece, class_aece_lst, bin_bound = aece_criterion_seg(logits, labels)
            print(aece.item())
            print(class_aece_lst)
            print('*******************')
            print(bin_bound) 
            
        # print('*****************************')
        # # mean_acc = Tp/totalp 
        # print('mean_acc: ', mean_acc)
        # print('*****************************')

        # iou = inter/union
        # acc = inter/preds
        # mIoU = iou.mean().item()
        # mAcc = acc.mean().item()
        # print_iou(iou, acc, mIoU, mAcc)    

        return iou, mIoU, acc, mAcc

def makecalipredgt(model, testloader, args, da_model, lightnet, weights):
    interp = nn.Upsample(size=(1080,1920), mode='bilinear', align_corners=True)   #dark_zurich -> (1080,1920) 
    # print('******************') 
    model = model.eval() 
    all_pred = []
    all_true = []
    
    with torch.no_grad(): 
        for index, batch in tqdm(enumerate(testloader)): 
            image, label, edge, _, name = batch 
            da_model = da_model.eval()
            lightnet = lightnet.eval() 
            with torch.no_grad():
                r = lightnet(image.cuda())
                enhancement = image.cuda() + r
                if model == 'RefineNet':
                    output2 = da_model(enhancement)
                else:
                    _, output2 = da_model(enhancement)

            weights_prob = weights.expand(output2.size()[0], output2.size()[3], output2.size()[2], 19)
            weights_prob = weights_prob.transpose(1, 3)
            output2 = output2 * weights_prob
            output2 = interp(output2).cpu().numpy() 
            pred = torch.tensor(output2) 

            # print(pred.shape)
            pred = F.softmax(pred, dim=1) ## for normalising prediction values 

            nc = pred.shape[1]  
            # pred_label = torch.argmax(pred, dim=1)  
            (pred_conf, pred_label) = torch.max(pred,dim=1)  
            # print(np.unique(pred_label))
            # print(np.unique(label)) 
            # print(np.unique(pred_conf)) ## now it is under 0 and 1
            label = (label == pred_label) 
            # label = F.one_hot(label, num_classes=nc) ## not req 

            pred= pred_conf.squeeze()
            label = label.squeeze()
            # print(pred.shape)  # torch.Size([1080, 1920])
            # print(label.shape) # torch.Size([1080, 1920]) 
            pred = np.array(pred)
            pred = pred.flatten() 
            # print(np.unique(pred)) 
            label = np.array(label).astype(int) 
            # print(label)
            label = label.flatten()

            # print(np.unique(label)) # wrong ..true false coming ...now it is right conversion from bool to integer 
            all_pred.append(pred)  
            all_true.append(label)  
             
            
    # all_pred = np.vstack(all_pred)
    # all_true = np.vstack(all_true)    

    all_pred = np.hstack(all_pred) ## it works !!!
    all_true = np.hstack(all_true)  

    # print(all_pred.shape)  # (50, 2073600) 
    # print(all_true.shape) # (50, 2073600)
    # print('roger that') 
    return all_pred, all_true
    # return pred, label # its working 


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
        19: [0,  0, 0]
        }
    # # with open('./dataset/cityscapes_list/info.json') as f:
    #     data = json.load(f)

    # label_to_color = {
    #     0: [0, 0, 0],
    #     1: [255,255,255]
    # }

    img_height, img_width = img.shape

    img_color = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    for row in range(img_height):
        for col in range(img_width):
            label = img[row][col]
            img_color[row, col] = np.array(label_to_color[label])
            # img_color[row][col] = np.asarray(data['palette'][label])
    return img_color


def save_fake(model, testloader):
    model = model.eval()
    # interp = nn.Upsample(size=(1024,2048), mode='bilinear', align_corners=True)
    with torch.no_grad():
        for index, batch in tqdm(enumerate(testloader)):
            image, label, edge, _, name = batch
            output =  model(image.cuda())
            # output = interp(output).squeeze()
            output = output.squeeze()
            name = name[0].split("/")[-1]
            # print(name)
            save_pred(output, '../scratch/saved_models/CCM/save/result/rf_city_dzval', name)
    return


def save_pred(pred, direc, name):
    # palette = get_palette(256)
    pred = pred.cpu().numpy()
    # print(pred.shape)
    # pred = np.asarray(np.argmax(pred, axis=0), dtype=np.uint8)   ##### original
    # pred = np.asarray(np.argsort(pred, axis= 0)[-2], dtype = np.uint8)  ############ 2nd best prediction
    
    # if thresholding for binary segmentation 
    # pred[pred<0.5] = 0
    # pred[pred>=0.5] = 1  
    
    # pred = np.asarray(np.argmax(pred, axis=0))
    # print(pred.shape)
    label_img_color = label_img_to_color(pred)
    # print(label_img_color.dtype)
    # print(label_img_color.shape)
    # print(np.unique(label_img_color))
    # cv2.imwrite(osp.join(direc,name), label_img_color)
    im = Image.fromarray(label_img_color) # use always PIL or other library .. try to avoid cv2.. but if no other option then ok
    # im.save(osp.join(direc,name))
    im.save(osp.join(direc,name))
    return 
    # print('img saved!')

    # img = np.zeros((pred.shape[0],pred.shape[1],3))

    # print(np.unique(pred))
    # print(pred.shape)
    # print(plabel.shape)
    # print(img.shape)
    # cv2.imwrite("pred.png", img)
    # output_im = Image.fromarray(pred)
    # output_im.putpalette(palette)
    # output_im.save('pred.png')
         
    # print((plabel==i).nonzero()) #work 
    # print(plabel==i)

    # img = torch.tensor(img)
    # img = img.permute(2, 0, 1).unsqueeze(dim = 0)
    # print(img.shape) 
    # torchvision.utils.save_image(img,osp.join(direc,name)) 

    # img = img.reshape(3,pred.shape[0],-1)
    # print(img.shape)
    # print(np.unique(img))
    # unique, counts = np.unique(img, return_counts=True)
    # print(dict(zip(unique, counts)))
    # img = Image.fromarray((img * 255).astype(np.uint8))
    # img.save(osp.join(direc,name))


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    args = get_arguments()
    with open('./config/config.yml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = edict(cfg)
    # cfg.num_classes=args.num_classes
    # cfg.num_channels = args.num_channels
    # if args.single:
    #from model.fuse_deeplabv2 import Res_Deeplab
    if args.model=='deeplab':
        # print('*********************************')
        model = Res_Deeplab(num_classes=args.num_classes).cuda()
    elif args.model == 'unet':
        # print(')))))))))))))))))))))))))))')
        # model = UNet(n_class=2).cuda() 
        # model = UNet_mod(n_class=2).cuda()  
        model = UNet_mod(n_channels=args.num_channels, n_class = args.num_classes, fake_ce = cfg.fake_ce).cuda()
    else:
        model = FCN8s(num_classes = args.num_classes).cuda() 

    # model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.frm)) #original 
    # model.load_state_dict(torch.load(args.frm,strict=False))

    # original saved file with DataParallel
    # state_dict = torch.load(args.frm)
    # # create new OrderedDict that does not contain `module.`
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:] # remove `module.`
    #     new_state_dict[name] = v
    # # load params
    # model.load_state_dict(new_state_dict)

    model.eval().cuda() 
    # print(model)
    testloader  = init_test_dataset(cfg, args.dataset, set='val')
    # print(len(testloader))
    # print('****************************************************')
    # save_fake(model, testloader)
    
    da_model, lightnet, weights = pred(cfg.num_classes, cfg.model_dannet, cfg.restore_from_da, cfg.restore_light_path)
    fake_ce = cfg.fake_ce

    # if args.calibrationcalc:
    #     all_pred, all_true = makecalipredgt(model, testloader, args, da_model, lightnet, weights)
    #     dict_metric = fast_ece(all_true, all_pred)  
    #     print(dict_metric)

    # else:
        # iou, mIoU, acc, mAcc = compute_iou(model, testloader, args, da_model, lightnet, weights, fake_ce) # original 
    
    iou, mIoU, acc, mAcc = compute_iou(model, testloader, args, da_model, lightnet, weights, fake_ce) # original
    return

    # sys.stdout = Logger(osp.join(cfg['result'], args.frm+'.txt'))

    # best_miou = 0.0
    # best_iter = 0
    # best_iou = np.zeros((args.num_classes, 1))

   
    # for i in range(args.start, 25):
    #     model_path = osp.join(cfg['snapshot'], args.frm, 'GTA5_{0:d}.pth'.format(i*2000))# './snapshots/GTA2Cityscapes/source_only/GTA5_{0:d}.pth'.format(i*2000)
    #     model = Res_Deeplab(num_classes=args.num_classes)
    #     #model = nn.DataParallel(model)

    #     model.load_state_dict(torch.load(model_path))
    #     model.eval().cuda()
    #     testloader = init_test_dataset(cfg, args.dataset, set='train') 

    #     iou, mIoU, acc, mAcc = compute_iou(model, testloader)

    #     print('Iter {}  finished, mIoU is {:.2%}'.format(i*2000, mIoU))

if __name__ == '__main__':
    main()