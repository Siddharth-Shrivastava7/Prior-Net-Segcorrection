# commenting the original btad for unet based real/fake 
from numpy.core.numeric import indices
import torch
from utils.optimize import *
from .base_trainer import BaseTrainer
from pytorch_memlab import profile
from easydict import EasyDict as edict
import os.path as osp
from dataset import dataset
import  torch.optim as optim
from tqdm import tqdm
# import neptune
import math
from PIL import Image
import copy
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import torch.nn as nn
from dataset.network.dannet_pred import pred

class Trainer(BaseTrainer):
    def __init__(self, model, config, writer):
        self.model = model

        self.config = config
        self.writer = writer

        ### dannet model load...
        # sv = torch.load('/home/sidd_s/scratch/saved_models_hpc/saved_models/DANNet/dannet_psp.pth')
        self.da_model, self.lightnet, self.weights = pred(self.config.num_classes, self.config.model_dannet, self.config.restore_from_da, self.config.restore_light_path)
        # print('**************')
        
    def iter(self, batch):
        
        img, seg_label, _, _, name = batch
        # print(img.shape) 
        interp = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)
        ### Dannet pred..to get predictions 
        model = self.da_model.eval()
        lightnet = self.lightnet.eval()
        weights = self.weights

        with torch.no_grad():
            r = lightnet(img.cuda())
            enhancement = img.cuda() + r
            if model == 'RefineNet':
                output2 = model(enhancement)
            else:
                _, output2 = model(enhancement)

        weights_prob = weights.expand(output2.size()[0], output2.size()[3], output2.size()[2], 19)
        weights_prob = weights_prob.transpose(1, 3)
        output2 = output2 * weights_prob
        # img = output2 
        # print(output2.shape) # torch.Size([4, 19, 65, 65]) 
        output2 = interp(output2).cpu().numpy() ## we have to upsample it... 
        # print(output2.shape) # (4, 19, 512, 512) 
        # print('*****************') 
        img = torch.tensor(output2)        

        ## making it probability distribution
        # img_prob = F.softmax(img.cuda(), dim=1)  
        # print(torch.max(img_prob)) 
        # print(torch.min(img_prob))

        # print(torch.unique(seg_label))  
        # print(img)  
        # print(torch.max(img))  # tensor(14.3076) for one of the input
        # print(torch.min(img)) # tensor(-40.9996) for one of the input
        ### Dannet pred... 

        if self.config.one_hot: 
            ## one_hot_conversion.... 
            # print('&&&&&&&&&&&&&&&&&&&')
            nc = img.shape[1]
            img = torch.argmax(img, dim=1)
            img = F.one_hot(img, num_classes=nc)
            img = img.transpose(3,2).transpose(2,1) 
            # one_hot_conversion....
            seg_pred = self.model(img.float().cuda())  ## one hot input
        else:
            # print('********************')
            # seg_pred = self.model(F.softmax(img.cuda(), dim=1)) ## mod.....resize@@@@
            # seg_pred = self.model(img_prob.cuda()) 
            seg_pred = self.model(img.cuda()) 
        
        ## making it probability distribution 
        # seg_pred_prob = F.softmax(seg_pred, dim=1)

        # print(seg_pred.shape) # torch.Size([12, 19, 512, 512])
        # print(torch.max(seg_pred)) 
        # print(torch.min(seg_pred))
        seg_label = seg_label.long().cuda() # cross entropy   
        # b, c, h, w = img.shape
        # print(img.shape) # torch.Size([16, 19, 512, 512])
        # print(torch.unique(seg_label)) # tensor([255], device='cuda:0') wrong brother
        
        # img = F.softmax(img, dim=1) ## extra see ...not giving not results not optimising early as comparre to just logits....but then 

        # print(seg_pred.shape) # torch.Size([16, 19, 512, 512])
        # print('********')
        # print(seg_label.shape) 
        # seg_pred = nn.DataParallel(self.model(img.cuda())) 
        # print('yes')
        # print(seg_label.shape, seg_pred.shape) # torch.Size([16, 512, 512]) torch.Size([16, 19, 512, 512])
        # print(seg_label.shape)
        # print('&&&&&&&&&&&&&&&&&&&&&&&&')
        # print(img.shape) # torch.Size([16, 19, 512, 512])
        # print(seg_pred.shape)
        # print('**********************')
        loss = CrossEntropy2d() # ce loss 
        # print('********************')

        if self.config.fake_ce and self.config.real_en: 
            loss2 = entropymax()
            # print('yo') 
            print('<<<<<<<<<<<<<<<<<<<<<<<<<<')
            # print(seg_pred.shape) # torch.Size([12, 19, 512, 512])
            seg_loss = loss(seg_pred, seg_label) + loss2(seg_pred, seg_label)             
        
        # if self.config.fake_ce and not self.config.real_en:
        #     # print('***********')
        #     # print('>>>>>>>>>>>>>>>>>>>>>')
        #     # entropy_map = -torch.sum(seg_pred*torch.log(seg_pred), dim=1) 
        #     ## weighting entropy map for the real in our model's output
        #     seg_loss = loss(seg_pred, seg_label)
        #     # print('**********')

        else: 
            # print('*************************************')
            # seg_pred = F.softmax(seg_pred, dim=1) ## cause the input tensor is also being under proba distribtution...so making it also...will be using.. NLL loss in cross entropy  ## not using 

            ##############################################################################################
            ### an exp.. to perfrom...which will be more related to probability (MAP) case of our hypothesis
            # seg_pred = F.softmax(seg_pred, dim=1) ## proba distri for the prior 
            ## now the tensor what we got from dannet assuming it to be prob distri only...but it has negative values tooo...but since over the same the argmax was taken thus that's our proba distri ... no more changes...now the posterior will be...as below defined....       that posterior will be the logits or the proba that we need to do...but my guess it should be the proba (still or exp with both)        
            
            ### an exp...
            ##############################################################################################

            ###### posterior = prior (seg pred...i.e. prior which is getting refine) * likelihood (orginal tensor)
            # post = seg_pred * img.cuda().detach() ## detach is used to remove the grad calc for the original tensor...such that while backprop it doesn't update only the prior adjusts itself to lower the loss in the coming epochs and thus to reduce loss significantly.  ####### This is not possible when we are providing the input as 3 channel pred image or one hot encoded 19 channel image since then at the posterior time ..the likelihood will be 19 channel one hot thing 
            
            # post = seg_pred_prob * img_prob.cuda().detach()
            # post1 = seg_pred * img.cuda().detach()   

            ### repeat 
            # seg_pred2 = self.model(post1.cuda()) 
            # post = seg_pred2 * post1.cuda()  

            # print(torch.argmax(post, dim=1))  
            
            img_prob = F.softmax(img.cuda(), dim=1)  
            entropy_map = -torch.sum(img_prob*torch.log(img_prob), dim=1) 
            # print(torch.max(entropy_map))   
            entropy_map_norm = entropy_map / torch.max(entropy_map) 
            # print(entropy_map_norm.shape) # torch.Size([12, 512, 512])
            entropy_map_norm = entropy_map_norm.unsqueeze(dim=1)  

            # ind = torch.argmax(post, dim=1) 
            # # print(ind) 
            # post[:, ind,:,:] = 1

            # print(post)

            # print(post) 
            # print('******************') 
            # print(img)  
            # print('****************')
            # print(entropy_map_norm)

            # out = post.cuda() * entropy_map_norm.detach()  + (1-entropy_map_norm.detach()) * img.cuda().detach()  
            out = seg_pred.cuda() * entropy_map_norm.detach()  + (1-entropy_map_norm.detach()) * img.cuda().detach() 

            # # repeat 
            seg_pred2 = self.model(out.cuda())   
            out2 = seg_pred2.cuda() * entropy_map_norm.detach()  + (1-entropy_map_norm.detach()) * img.cuda().detach()  

            ## repeat 
            seg_pred3 = self.model(out2.cuda())   
            out3 = seg_pred3.cuda() * entropy_map_norm.detach()  + (1-entropy_map_norm.detach()) * img.cuda().detach() 


            # print(post.shape) # torch.Size([12, 19, 512, 512]) 
            # print(post)
            # post = F.softmax(post, dim=1) ## not using   
            # print(post) 
            # post = F.log_softmax(post, dim=1)  ## can use this with NLL pytorch function ..yes yes 
            # post = F.log_softmax(seg_pred,dim=1) + F.log_softmax(img.cuda().detach(), dim=1)  ## exp ...let's see ..not works..and not neeeded too
            # seg_loss = loss(seg_pred, seg_label) ## original
            # seg_loss = loss(post, seg_label)  # posterior MAP estimate 
            # seg_loss = loss(out2, seg_label) 
            seg_loss = loss(out3, seg_label)
    
        self.losses.seg_loss = seg_loss
        loss = seg_loss  
        loss.backward()      

    def train(self):
        writer = SummaryWriter(comment="unet_e2e_acdc_ent_iterate_iterate_drp_bt4")

        # if self.config.neptune:
        #     neptune.init(project_qualified_name='solacex/segmentation-DA')
        #     neptune.create_experiment(params=self.config, name=self.config['note'])

        if self.config.multigpu:       
            self.optim = optim.SGD(self.model.module.optim_parameters(self.config.learning_rate),
                          lr=self.config.learning_rate, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        else:
            # self.optim = optim.SGD(self.model.optim_parameters(self.config.learning_rate),
            #               lr=self.config.learning_rate, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
            self.optim = optim.SGD(self.model.parameters(),
                          lr=self.config.learning_rate, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
            # self.optim = optim.Adam(self.model.parameters(),
            #             lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        
        # self.loader = dataset.init_test_dataset(self.config, self.config.target, set='val') # just for testing 
        self.loader, _ = dataset.init_source_dataset(self.config) #, source_list=self.config.src_list)

        cu_iter = 0
        early_stop_patience = 0
        best_val_epoch_loss = float('inf') 
        train_epoch_loss = 0
        print('Lets ride...')

        for epoch in range(self.config.epochs):
            epoch_infor = ('epoch = {:6d}/{:6d}, exp = {}'.format(epoch, int(self.config.epochs), self.config.note))
            
            print(epoch_infor)
            # print(len(self.loader))
            # print(self.config.learning_rate)
 
            for i_iter, batch in tqdm(enumerate(self.loader)):
                cu_iter +=1
                adjust_learning_rate(self.optim, cu_iter, self.config)
                self.optim.zero_grad()
                self.losses = edict({})
                losses = self.iter(batch)

                ## print(self.config['model'])
                # print(self.losses['seg_loss'].item())
                # print(cu_iter)
                train_epoch_loss += self.losses['seg_loss'].item()
                print('train_iter_loss:', self.losses['seg_loss'].item())
                # valid_epoch_loss = self.validatebtad(i_iter)
                # print('validation_loss: ', valid_epoch_loss)
                self.optim.step() 

            train_epoch_loss /= len(iter(self.loader)) 
            # if cu_iter % self.config.print_freq ==0:
            #     self.print_loss(cu_iter)
            # if self.config.val and cu_iter % self.config.val_freq ==0 and cu_iter!=0:
                # miou = self.validate() 
            if self.config.val:

                loss_infor = 'train loss :{:.4f}'.format(train_epoch_loss)
                print(loss_infor)

                valid_epoch_loss = self.validatebtad(epoch)

                writer.add_scalars('Epoch_loss',{'train_tensor_epoch_loss': train_epoch_loss,'valid_tensor_epoch_loss':valid_epoch_loss},epoch)  

                if(valid_epoch_loss < best_val_epoch_loss):
                    early_stop_patience = 0 ## early stopping variable 
                    best_val_epoch_loss = valid_epoch_loss
                    print('********************')
                    print('best_val_epoch_loss: ', best_val_epoch_loss)
                    print("MODEL UPDATED")
                    name = self.config['model'] + '.pth' # for the ce loss 
                    # name = 'acdc_tensor_focal.pth' # focal loss 
                    torch.save(self.model.state_dict(), osp.join(self.config["snapshot"], name))
                # else: ## early stopping with patience of 50
                #     early_stop_patience = early_stop_patience + 1 
                #     if early_stop_patience == 50:
                #         break
                self.model = self.model.train()
            
            # if early_stop_patience == 50: 
            #     print('Early_Stopping!!!')
            #     break ``

        writer.export_scalars_to_json("./all_scalars.json")
        writer.close()

        if self.config.neptune:
            neptune.stop()
                
    def resume(self):
        self.tea = copy.deepcopy(self.model)
        self.round_start = self.config.round_start #int(math.ceil(iter_num/self.config.num_steps) -1 )
        print('Resume from Round {}'.format(self.round_start))
        if self.config.lr_decay == 'sqrt':
            self.config.learning_rate = self.config.learning_rate/((math.sqrt(2))**self.round_start)

    def save_best(self, name):
        name = str(name)
        if 'pth' not in name:
            name = name +'.pth'
        torch.save(self.model.state_dict(), osp.join(self.config["snapshot"], name))
    def save_model(self, iter, rep_teacher=False):
        tmp_name = '_'.join((self.config.source, str(iter))) + '.pth'
        torch.save(self.model.state_dict(), osp.join(self.config['snapshot'], tmp_name))


class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255, real_label=100):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label
        self.real_label = real_label

    def forward(self, predict, target, weight=None):
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
        target_mask = (target >= 0) * (target != self.ignore_label) * (target!= self.real_label)  ### should not apply ce loss to the real labels (addon)
        target = target[target_mask]
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        # print(predict.shape)
        # print('***************')
        # print(target.shape)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average) 
        # loss = F.nll_loss(predict, target, weight=weight, size_average= self.size_average )  ## NLL loss cause the pred is now in softmax form..not using cause....nan is showing up sometimes    
        return loss

class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2, ignore_label=255):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma
        self.ignore_label = ignore_label

    def forward(self, inputs, targets):
        # n, h, w = inputs.size()
        targets_mask = (targets >= 0) * (targets != self.ignore_label)
        targets = targets[targets_mask]
        inputs = inputs[targets_mask]
        BCE_loss = F.binary_cross_entropy(inputs, targets)
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        # print(at[targets==1])
        # # print(at)
        # print('**********')
        pt = torch.exp(-BCE_loss)
        # print(pt)
        # print('^^^^^^^^^^^^')
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

class entropymax(nn.Module): 
    def __init__(self, real_label = 100):
        super(entropymax, self).__init__()
        self.real_label = real_label
    
    def forward(self, inputs, targets):
        ## mask creation.. 
        target_mask = (targets == self.real_label)  ## binary mask for the region where the actual real is there
        ## mask creation..
        n, c, h, w = inputs.size()
        inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous() ## input shape --> (n, h, w, c)
        # inputs = inputs[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        inputs = inputs[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        # print(inputs.shape)
        # print(inputs.shape)  # torch.Size([0]) ....wrong brother
        # inputs = inputs + 1e-16 # for numerical stability while taking log 
        output = F.softmax(inputs, dim=1) * F.log_softmax(inputs, dim=1)  
        loss = torch.mean(torch.sum(output, dim=1))  ## just loss = output.sum()
        return loss     
