import torch 
import numpy as np 
import math  
import torch.nn as nn   
from torch.nn import functional as F
from tqdm import tqdm

def fast_ece(y_true, y_pred, n_bins = 10):
    ## ~sklearn code
    bins = np.linspace(0., 1.- 1./n_bins, n_bins) # alles >= laatste waarde word in extra bin gestoken, dus daarom deze rare notatie
    binids = np.digitize(y_pred, bins) - 1  ## yield indices default--> bins[i-1] <= x < bins[i]  ## binids is the problem...
    # print(binids.shape) # (50, 2073600)  ## (103680000,) it works!! (ie. conversion to one d array helps)

    bin_sums = np.bincount(binids, weights=y_pred, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins)) 
    # bin_total = np.bincount(binids) ## didn't work either 

    nonzero = bin_total != 0 # don't use empty bins
    prob_true = (bin_true[nonzero] / bin_total[nonzero]) # acc
    prob_pred = (bin_sums[nonzero] / bin_total[nonzero]) # conf

    weights = bin_total[nonzero] / np.sum(bin_total[nonzero])
    l1 = np.abs(prob_true-prob_pred)
    ece = np.sum(weights*l1)
    mce = l1.max()
    l1 = l1.sum()
    return {"acc": prob_true, "conf": prob_pred,"ECE": ece, "MCE": mce, "l1": l1}


class ClasswiseECELossSeg(nn.Module):
    '''
    Compute Classwise ECE or static calibration error for segmentation
    '''
    def __init__(self, n_bins=15):
        super(ClasswiseECELossSeg, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        self.class_sce_lst = []

    def forward(self, logits, labels):
        num_classes = logits.shape[1]  
        # print(num_classes)
        softmaxes = F.softmax(logits, dim=1)  
        per_class_sce = None  
        class_sce_lst = [[] for i in range(num_classes)]

        for i in range(num_classes):  
            class_confidences = softmaxes[:, i, :, :] 
            # print(class_confidences.shape)  # torch.Size([50, 1080, 1920]) 
            class_sce = torch.zeros(1, device=logits.device)
            labels_in_class = labels.eq(i) 
            
            # print(labels_in_class.shape)  # torch.Size([50, 1080, 1920]) 
            # print(i) 
            # print(torch.unique(labels_in_class)) # tensor([False,  True], device='cuda:0')
            # print('**************')
            for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
                in_bin = class_confidences.gt(bin_lower.item()) * class_confidences.le(bin_upper.item())
                prop_in_bin = in_bin.float().mean()
                if prop_in_bin.item() > 0:
                    accuracy_in_bin = labels_in_class[in_bin].float().mean()
                    avg_confidence_in_bin = class_confidences[in_bin].mean() 
                    class_sce_lst[i].append((accuracy_in_bin.item(), avg_confidence_in_bin.item()))
                    class_sce += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            if (i == 0):
                per_class_sce = class_sce
            else: 
                per_class_sce = torch.cat((per_class_sce, class_sce), dim=0) 
        sce = torch.mean(per_class_sce) 
        return sce, class_sce_lst


class ClasswiseAdaptiveECELoss(nn.Module):
    '''
    Compute Classwise Adaptive ECE
    '''
    def __init__(self, n_bins=15):
        super(ClasswiseAdaptiveECELoss, self).__init__()
        self.nbins = n_bins

    def histedges_equalN(self, x):
        npt = len(x)
        return np.interp(np.linspace(0, npt, self.nbins + 1),
                     np.arange(npt),
                     np.sort(x))
    ## The above function should change for histogram for seg	..I think we could flatten that and still get the same value..

    def forward(self, logits, labels):
        num_classes = logits.shape[1]   
        softmaxes = F.softmax(logits, dim=1)
        # confidences, predictions = torch.max(softmaxes, 1)
        per_class_aece = None  
        class_aece_lst = [[] for i in range(num_classes)]
        bin_bound = [[] for i in range(num_classes)]

        for i in tqdm(range(num_classes)):
            class_confidences = softmaxes[:, i, :, :]  
            class_aece = torch.zeros(1, device=logits.device) 
            labels_in_class = labels.eq(i) 
            n, bin_boundaries = np.histogram(class_confidences.flatten().cpu().detach(), self.histedges_equalN(class_confidences.flatten().cpu().detach()))
            #print(n,confidences,bin_boundaries)
            self.bin_lowers = bin_boundaries[:-1]
            self.bin_uppers = bin_boundaries[1:]
            ece = torch.zeros(1, device=logits.device)
            for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
                # Calculated |confidence - accuracy| in each bin
                in_bin = class_confidences.gt(bin_lower.item()) * class_confidences.le(bin_upper.item())
                prop_in_bin = in_bin.float().mean()
                if prop_in_bin.item() > 0:
                    accuracy_in_bin = labels_in_class[in_bin].float().mean()
                    avg_confidence_in_bin = class_confidences[in_bin].mean()
                    class_aece_lst[i].append((accuracy_in_bin.item(), avg_confidence_in_bin.item()))
                    # bin_bound[i].append((bin_lower, bin_upper))
                    class_aece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            if (i == 0):
                per_class_aece = class_aece
            else: 
                per_class_aece = torch.cat((per_class_aece, class_aece), dim=0)
        aece = torch.mean(per_class_aece)
        return aece, class_aece_lst 