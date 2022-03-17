import torch
import os.path as osp
# import neptune
import torch.nn as nn
# import neptune 
from dataset import dataset
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from dataset.network.dannet_pred import pred 
class BaseTrainer(object):
    def __init__(self, models, optimizers, loaders, up_s, up_t, config,  writer):
        self.model = models
        self.optim = optimizers
        self.loader = loaders
        self.config = config
        self.output = {}
        self.up_src = up_s
        self.up_tgt = up_t
        self.writer = writer
        self.da_model, self.lightnet, self.weights = pred(self.config.num_classes, self.config.model_dannet, self.config.restore_from_da, self.config.restore_light_path)

    def forward(self):
        pass
    def backward(self):
        pass

    def iter(self):
        pass

        
    def train(self):
        for i_iter in range(self.config.num_steps):
            losses = self.iter(i_iter)
            if i_iter==0 and self.config.neptune:
                neptune.init(project_qualified_name='solacex/segmentation-DA')
                neptune.create_experiment(params=self.config, name=self.config['note'])
            if i_iter % self.config.print_freq ==0:
                self.print_loss(i_iter)
            if i_iter % self.config.save_freq ==0 and i_iter != 0:
                self.save_model(i_iter)
            if self.config.val and i_iter % self.config.val_freq ==0 and i_iter!=0:
                self.validate()
        neptune.stop()

    def save_model(self, iter):
        tmp_name = '_'.join((self.config.source, str(iter))) + '.pth'
        torch.save(self.model.state_dict(), osp.join(self.config['snapshot'], tmp_name))

    def print_loss(self, iter):
        iter_infor = ('iter = {:6d}/{:6d}, exp = {}'.format(iter, self.config.num_steps, self.config.note))
        to_print = ['{}:{:.4f}'.format(key, self.losses[key].item()) for key in self.losses.keys()]
        loss_infor = '  '.join(to_print)
        if self.config.screen:
            print(iter_infor +'  '+ loss_infor)
        if self.config.neptune:
            for key in self.losses.keys():
                neptune.send_metric(key, self.losses[key].item())
        if self.config.tensorboard and self.writer is not None:
            for key in self.losses.keys():
                self.writer.add_scalar('train/'+key, self.losses[key], iter)

    def validate(self):
        self.model = self.model.eval()
        testloader = dataset.init_test_dataset(self.config, self.config.target_val, set='val')
        interp = nn.Upsample(size=(1080, 1920), mode='bilinear', align_corners=True)
        union = torch.zeros(self.config.num_classes, 1,dtype=torch.float).cuda().float()
        inter = torch.zeros(self.config.num_classes, 1, dtype=torch.float).cuda().float()
        preds = torch.zeros(self.config.num_classes, 1, dtype=torch.float).cuda().float()
        with torch.no_grad():
            for index, batch in tqdm(enumerate(testloader)):
                image, label, _, _, name = batch
                output =  self.model(image.cuda())
                label = label.cuda()
                output = interp(output).squeeze()
                C, H, W = output.shape
                Mask = (label.squeeze())<C

                pred_e = torch.linspace(0,C-1, steps=C).view(C, 1, 1)
                pred_e = pred_e.repeat(1, H, W).cuda()
                pred = output.argmax(dim=0).float()
                pred_mask = torch.eq(pred_e, pred).byte()
                pred_mask = pred_mask*Mask.byte()

                label_e = torch.linspace(0,C-1, steps=C).view(C, 1, 1)
                label_e = label_e.repeat(1, H, W).cuda()
                label = label.view(1, H, W)
                label_mask = torch.eq(label_e, label.float()).byte()
                label_mask = label_mask*Mask.byte()

                tmp_inter = label_mask+pred_mask.byte()
                cu_inter = (tmp_inter==2).view(C, -1).sum(dim=1, keepdim=True).float()
                cu_union = (tmp_inter>0).view(C, -1).sum(dim=1, keepdim=True).float()
                cu_preds = pred_mask.view(C, -1).sum(dim=1, keepdim=True).float()

                union+=cu_union
                inter+=cu_inter
                preds+=cu_preds

            iou = inter/union
            acc = inter/preds
            if C==16:
                iou = iou.squeeze()
                class13_iou = torch.cat((iou[:3], iou[6:]))
                class13_miou = class13_iou.mean().item()
                print('13-Class mIoU:{:.2%}'.format(class13_miou))
            mIoU = iou.mean().item()
            mAcc = acc.mean().item()
            iou = iou.cpu().numpy()
            print('mIoU: {:.2%} mAcc : {:.2%} '.format(mIoU, mAcc))
            if self.config.neptune:
                neptune.send_metric('mIoU', mIoU)
                neptune.send_metric('mAcc', mAcc)
        return mIoU

    
    def validate2(self, count):
        self.model = self.model.eval()
        total_loss = 0
        testloader = dataset.init_test_dataset(self.config, self.config.target_val, set='val')
        interp = nn.Upsample(size=(1080,1920), mode='bilinear', align_corners=True)
        for i_iter, batch in tqdm(enumerate(testloader)):
            img, seg_label, _, _, name = batch
            seg_label = seg_label.long().cuda()
            b, c, h, w = img.shape
            # print(img.shape)
            seg_pred = self.model.forward(img.cuda())
            # print(seg_pred.shape, seg_label.shape)
            seg_pred = interp(seg_pred)
            seg_loss = F.cross_entropy(seg_pred, seg_label, ignore_index=255)
            total_loss += seg_loss.item()
        total_loss /= len(iter(testloader))
        print('---------------------')
        print('Validation seg loss: {} at iter {}'.format(total_loss,count))
        return total_loss
    
    # def validatebtad(self, count):
    #     self.model = self.model.eval()
    #     total_loss = 0
    #     testloader = dataset.init_test_datasetbtad(self.config, self.config.target, set='val')
    #     interp = nn.Upsample(size=(1080,1920), mode='bilinear', align_corners=True)
    #     union = torch.zeros(self.config.num_classes, 1,dtype=torch.float).cuda().float()
    #     inter = torch.zeros(self.config.num_classes, 1, dtype=torch.float).cuda().float()
    #     preds = torch.zeros(self.config.num_classes, 1, dtype=torch.float).cuda().float()
    #     with torch.no_grad():
    #         for index, batch in tqdm(enumerate(testloader)):
    #             image, label, _, _, name = batch
    #             output =  self.model(image.cuda())
    #             label = label.cuda()
    #             output = interp(output).squeeze()
    #             C, H, W = output.shape
    #             Mask = (label.squeeze())<C

    #             pred_e = torch.linspace(0,C-1, steps=C).view(C, 1, 1)
    #             pred_e = pred_e.repeat(1, H, W).cuda()
    #             pred = output.argmax(dim=0).float()
    #             pred_mask = torch.eq(pred_e, pred).byte()
    #             pred_mask = pred_mask*Mask.byte()

    #             label_e = torch.linspace(0,C-1, steps=C).view(C, 1, 1)
    #             label_e = label_e.repeat(1, H, W).cuda()
    #             label = label.view(1, H, W)
    #             label_mask = torch.eq(label_e, label.float()).byte()
    #             label_mask = label_mask*Mask.byte()

    #             tmp_inter = label_mask+pred_mask.byte()
    #             cu_inter = (tmp_inter==2).view(C, -1).sum(dim=1, keepdim=True).float()
    #             cu_union = (tmp_inter>0).view(C, -1).sum(dim=1, keepdim=True).float()
    #             cu_preds = pred_mask.view(C, -1).sum(dim=1, keepdim=True).float()

    #             union+=cu_union
    #             inter+=cu_inter
    #             preds+=cu_preds

    #         iou = inter/union
    #         acc = inter/preds
    #         if C==16:
    #             iou = iou.squeeze()
    #             class13_iou = torch.cat((iou[:3], iou[6:]))
    #             class13_miou = class13_iou.mean().item()
    #             print('13-Class mIoU:{:.2%}'.format(class13_miou))
    #         mIoU = iou.mean().item()
    #         mAcc = acc.mean().item()
    #         iou = iou.cpu().numpy()
    #         print('mIoU: {:.2%} mAcc : {:.2%} '.format(mIoU, mAcc))
    #         if self.config.neptune:
    #             neptune.send_metric('mIoU', mIoU)
    #             neptune.send_metric('mAcc', mAcc)
                
    #     for i_iter, batch in tqdm(enumerate(testloader)):
    #         img, seg_label, _, _, name = batch
    #         seg_label = seg_label.long().cuda()
    #         b, c, h, w = img.shape
    #         # print(img.shape)
    #         seg_pred = self.model(img.cuda())
    #         # print(seg_pred.shape, seg_label.shape)
    #         seg_pred = interp(seg_pred)
    #         seg_loss = F.cross_entropy(seg_pred, seg_label, ignore_index=255)
    #         total_loss += seg_loss.item()
    #     total_loss /= len(iter(testloader))
    #     print('---------------------')
    #     print('Validation seg loss: {} at iter {}'.format(total_loss,count))
    #     return total_loss

    def validatebtad(self, epoch):
        self.model = self.model.eval() 
        total_loss = 0
        testloader = dataset.init_test_dataset(self.config, self.config.target, set='val')
        for i_iter, batch in tqdm(enumerate(testloader)):
            img, seg_label, _, _, name = batch
            # print(img.shape)
            # print('******')
            # print(seg_label.shape)
            seg_label = seg_label.long().cuda() # ce loss 
            b, c, h, w = img.shape
            # print(img.shape)
            # print('********')
            interp = nn.Upsample(size=(1080, 1920), mode='bilinear', align_corners=True)  
            # original eval 
            # interp = nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True) 
            ## dannet pred..to get tensor predictions 

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
            ## dannet pred 
            if self.config.one_hot: 
                ## one_hot_conversion.... 
                nc = img.shape[1]
                img = torch.argmax(img, dim=1)
                img = F.one_hot(img, num_classes=nc)
                img = img.transpose(3,2).transpose(2,1) 
                # one_hot_conversion....
                seg_pred = self.model(img.float().cuda())  ## one hot input
            else:
                seg_pred = self.model(F.softmax(img.cuda(), dim=1)) ## mod.....resize@@@@             
            # seg_pred = interp(seg_pred).squeeze(dim=1)  # #original eval
            # print(seg_pred.shape, seg_label.shape)
            loss = CrossEntropy2d()

            if self.config.fake_ce and self.config.real_en: 
                loss2 = entropymax()
                seg_loss = loss(seg_pred, seg_label) + loss2(seg_pred, seg_label)        

            # elif self.config.fake_ce and not self.config.real_en:
            #     seg_loss = loss(seg_pred, seg_label)
            #     # print('**********')

            else: 
                # seg_pred = F.softmax(seg_pred, dim=1)  ## cause the input tensor is also being under proba distribtution...so making it also...will be using.. NLL loss in cross entropy ## not using 

                ##### posterior = prior (seg pred...i.e. prior which is getting refine) * likelihood (orginal tensor)
                post = seg_pred * img.cuda().detach() ## not going backwards so...no worry here not require to use detach
                # post = seg_pred * img.cuda() ## causing currently using pred label to gt label exp....
                
                img_prob = F.softmax(img.cuda(), dim=1)  
                entropy_map = -torch.sum(img_prob*torch.log(img_prob), dim=1) 
                # print(torch.max(entropy_map))   
                entropy_map_norm = entropy_map / torch.max(entropy_map) 
                # print(entropy_map_norm.shape) # torch.Size([12, 512, 512])
                entropy_map_norm = entropy_map_norm.unsqueeze(dim=1)

                # out = post.cuda() * entropy_map_norm.detach()  + (1-entropy_map_norm.detach()) * img.cuda().detach()
                out = seg_pred.cuda() * entropy_map_norm.detach()  + (1-entropy_map_norm.detach()) * img.cuda().detach()

                # post = F.softmax(post, dim=1) ## not using 
        
                # seg_loss = loss(seg_pred, seg_label) ## original
                # seg_loss = loss(post, seg_label) # posterior MAP estimate 
                
                seg_loss = loss(out, seg_label)

            total_loss += seg_loss.item()
        total_loss /= len(iter(testloader))
        print('---------------------')
        print('Validation seg loss: {} at epoch {}'.format(total_loss,epoch))
        return total_loss

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
        target_mask = (target >= 0) * (target != self.ignore_label) * (target!= self.real_label) 
        target = target[target_mask]
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        # print(predict.shape)
        # print('***************')
        # print(target.shape)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        # loss = F.nll_loss(torch.log(predict), target, weight=weight, size_average=self.size_average) ## NLL loss cause the pred is now in softmax form..
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