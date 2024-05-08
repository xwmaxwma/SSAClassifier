# Copyright (c) OpenMMLab. All rights reserved.
# Originally from https://github.com/visual-attention-network/segnext
# Licensed under the Apache License, Version 2.0 (the "License")
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmseg.ops import resize
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

from models.losses import SSALoss
import cv2
import numpy as np


class Matrix_Decomposition_2D_Base(nn.Module):
    """Base class of 2D Matrix Decomposition.

    Args:
        MD_S (int): The number of spatial coefficient in
            Matrix Decomposition, it may be used for calculation
            of the number of latent dimension D in Matrix
            Decomposition. Defaults: 1.
        MD_R (int): The number of latent dimension R in
            Matrix Decomposition. Defaults: 64.
        train_steps (int): The number of iteration steps in
            Multiplicative Update (MU) rule to solve Non-negative
            Matrix Factorization (NMF) in training. Defaults: 6.
        eval_steps (int): The number of iteration steps in
            Multiplicative Update (MU) rule to solve Non-negative
            Matrix Factorization (NMF) in evaluation. Defaults: 7.
        inv_t (int): Inverted multiple number to make coefficient
            smaller in softmax. Defaults: 100.
        rand_init (bool): Whether to initialize randomly.
            Defaults: True.
    """

    def __init__(self,
                 MD_S=1,
                 MD_R=64,
                 train_steps=6,
                 eval_steps=7,
                 inv_t=100,
                 rand_init=True):
        super().__init__()

        self.S = MD_S
        self.R = MD_R

        self.train_steps = train_steps
        self.eval_steps = eval_steps

        self.inv_t = inv_t

        self.rand_init = rand_init

    def _build_bases(self, B, S, D, R, cuda=False):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    def local_inference(self, x, bases):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x, return_bases=False):
        """Forward Function."""
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B * S, D, N)
        D = C // self.S
        N = H * W
        x = x.view(B * self.S, D, N)
        cuda = 'cuda' in str(x.device)
        if not self.rand_init and not hasattr(self, 'bases'):
            bases = self._build_bases(1, self.S, D, self.R, cuda=cuda)
            self.register_buffer('bases', bases)

        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R, cuda=cuda)
        else:
            bases = self.bases.repeat(B, 1, 1)

        bases, coef = self.local_inference(x, bases)

        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))

        # (B * S, D, N) -> (B, C, H, W)
        x = x.view(B, C, H, W)

        return x


class NMF2D(Matrix_Decomposition_2D_Base):
    """Non-negative Matrix Factorization (NMF) module.

    It is inherited from ``Matrix_Decomposition_2D_Base`` module.
    """

    def __init__(self, args=dict()):
        super().__init__(**args)

        self.inv_t = 1

    def _build_bases(self, B, S, D, R, cuda=False):
        """Build bases in initialization."""
        if cuda:
            bases = torch.rand((B * S, D, R)).cuda()
        else:
            bases = torch.rand((B * S, D, R))

        bases = F.normalize(bases, dim=1)

        return bases

    def local_step(self, x, bases, coef):
        """Local step in iteration to renew bases and coefficient."""
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # Multiplicative Update
        coef = coef * numerator / (denominator + 1e-6)

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        numerator = torch.bmm(x, coef)
        # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        # Multiplicative Update
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        """Compute coefficient."""
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ (B * S, D, R)^T @ (B * S, D, R) -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # multiplication update
        coef = coef * numerator / (denominator + 1e-6)

        return coef


class Hamburger(nn.Module):
    """Hamburger Module. It consists of one slice of "ham" (matrix
    decomposition) and two slices of "bread" (linear transformation).

    Args:
        ham_channels (int): Input and output channels of feature.
        ham_kwargs (dict): Config of matrix decomposition module.
        norm_cfg (dict | None): Config of norm layers.
    """

    def __init__(self,
                 ham_channels=512,
                 ham_kwargs=dict(),
                 norm_cfg=None,
                 **kwargs):
        super().__init__()

        self.ham_in = ConvModule(
            ham_channels, ham_channels, 1,norm_cfg=None, act_cfg=None)

        self.ham = NMF2D(ham_kwargs)

        self.ham_out = ConvModule(
            ham_channels, ham_channels, 1, norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x):
        enjoy = self.ham_in(x)
        enjoy = F.relu(enjoy, inplace=True)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        ham = F.relu(x + enjoy, inplace=True)

        return ham


@HEADS.register_module()
class LightHamHead_ssa(BaseDecodeHead):
    """SegNeXt decode head.

    This decode head is the implementation of `SegNeXt: Rethinking
    Convolutional Attention Design for Semantic
    Segmentation <https://arxiv.org/abs/2209.08575>`_.
    Inspiration from https://github.com/visual-attention-network/segnext.

    Specifically, LightHamHead is inspired by HamNet from
    `Is Attention Better Than Matrix Decomposition?
    <https://arxiv.org/abs/2109.04553>`.

    Args:
        ham_channels (int): input channels for Hamburger.
            Defaults: 512.
        ham_kwargs (int): kwagrs for Ham. Defaults: dict().
    """

    def __init__(self, ham_channels=512, ham_kwargs=dict(), **kwargs):
        super(LightHamHead_ssa, self).__init__(
            input_transform='multiple_select', **kwargs)
        self.ham_channels = ham_channels

        self.squeeze = ConvModule(
            sum(self.in_channels),
            self.ham_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
            )

        self.hamburger = Hamburger(ham_channels, ham_kwargs, **kwargs)

        self.align = ConvModule(
            self.ham_channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
            )
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)    
        self.get_feat_pos = nn.Conv2d(self.channels, self.channels, 3, 1, 1, bias=True, groups=self.channels)
        self.center_pos = nn.Embedding(self.num_classes, self.channels)
        
        self.center_content_proj =  nn.Sequential(
            nn.Linear(self.channels * 2, self.channels // 2, bias=False), 
            nn.ReLU(inplace=True),
            nn.Linear(self.channels // 2, self.channels),
        )   
        self.gt_center_content_proj =  nn.Sequential(
            nn.Linear(self.channels * 2, self.channels // 2, bias=False), 
            nn.ReLU(inplace=True),
            nn.Linear(self.channels // 2, self.channels),
        )   
        self.center_pos_proj = nn.Sequential(
            nn.Linear(self.channels * 2, self.channels // 2, bias=False), 
            nn.ReLU(inplace=True),
            nn.Linear(self.channels // 2, self.channels),
        )
        self.gt_center_pos_proj = nn.Sequential(
            nn.Linear(self.channels * 2, self.channels // 2, bias=False), 
            nn.ReLU(inplace=True),
            nn.Linear(self.channels // 2, self.channels),
        )
        
        self.center_proj = nn.Linear(self.channels, self.channels, bias=False)
        self.gt_center_proj = nn.Linear(self.channels, self.channels, bias=False)
        self.feat_proj = nn.Identity()
        self.ssa_loss = SSALoss(num_classes=self.num_classes)


    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)

        inputs = [
            resize(
                level,
                size=inputs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners) for level in inputs
        ]

        inputs = torch.cat(inputs, dim=1)
        # apply a conv block to squeeze feature map
        x = self.squeeze(inputs)
        # apply hamburger module
        x = self.hamburger(x)

        # apply a conv block to align feature map
        feat = self.align(x)
        output = self.cls_seg(feat)
        return output, feat
    
    def get_pred(self, x, proto):
        b, c, h, w = x.size()[:]
        if len(proto.shape[:]) == 3:
            # x: [b, c, h, w]
            # proto: [b, cls, c]  
            cls_num = proto.size(1)
            x = x / (torch.norm(x, 2, 1, True) + 1e-12)
            proto = proto / (torch.norm(proto, 2, -1, True) + 1e-12) # b, n, c
            x = x.contiguous().view(b, c, h*w)  # b, c, hw
            pred = proto @ x  # b, cls, hw
        elif len(proto.shape[:]) == 2:
            # x: [b, c, h, w]
            # proto: [cls, c]            
            cls_num = proto.size(0)
            x = x / (torch.norm(x, 2, 1, True)+ 1e-12)
            proto = proto / (torch.norm(proto, 2, 1, True)+ 1e-12)
            x = x.contiguous().view(b, c, h*w)  # b, c, hw
            proto = proto.unsqueeze(0)  # 1, cls, c
            pred = proto @ x  # b, cls, hw
        pred = pred.contiguous().view(b, cls_num, h, w)
        return pred * 15
    
    def get_gt_center(self, x, y, new_proto):
        h, w = x.shape[-2:]   
        y = F.interpolate(y.float(), size=(h, w), mode='nearest')  # b, 1, h, w
        unique_y = list(y.unique())
        if 255 in unique_y:
            unique_y.remove(255)

        for tmp_y in unique_y:
            tmp_mask = (y == tmp_y).float()
            tmp_proto = (x * tmp_mask).sum(0).sum(-1).sum(-1) / (tmp_mask.sum(0).sum(-1).sum(-1) + 1e-12)
            onehot_vec = torch.zeros(new_proto.shape[0], 1).cuda()  # cls, 1
            onehot_vec[tmp_y.long()] = 1
            new_proto = new_proto * (1 - onehot_vec) + tmp_proto.unsqueeze(0) * onehot_vec
            
        return new_proto

    def gt_ssa(self, x, y, new_proto, proto, feat_pos=None):
        b = x.size(0)
        raw_x = x.clone()
        
        new_proto = self.get_gt_center(x, y, new_proto)
        
        new_proto = torch.cat([new_proto, proto], -1)
        gt_proto = self.gt_center_content_proj(new_proto).unsqueeze(0).repeat(b, 1, 1)

        center_pos = self.center_pos.weight
        
        gt_center_pos_list = []       
        for i in range(b):
            gt_center_pos_list.append(self.get_gt_center(feat_pos[i].unsqueeze(0), y[i].unsqueeze(0), center_pos.detach().data))
        gt_center_pos = torch.stack(gt_center_pos_list, dim=0)
            
        gt_center_pos = self.gt_center_pos_proj(torch.cat([gt_center_pos, center_pos.unsqueeze(0).repeat(b, 1, 1)], dim=-1))
        
        new_center = self.gt_center_proj(self.with_pos_embed(gt_proto, gt_center_pos)) #(b, k, c)
        
        feat = self.feat_proj(self.with_pos_embed(raw_x, feat_pos))
        
        pred = self.get_pred(feat, new_center)
        return pred, gt_proto, gt_center_pos

    def pred_ssa(self, x, pred, proto, feat_pos, center_pos):    
        raw_x = x.clone()        
        b, c, h, w = raw_x.shape[:]
        pred = pred.view(b, proto.shape[0], h*w)
        pred = F.softmax(pred, 1)   # b, n, hw
        pred_proto = (pred @ raw_x.view(b, c, h*w).permute(0, 2, 1)) / (pred.sum(-1).unsqueeze(-1) + 1e-12)

        pred_proto = torch.cat([pred_proto, proto.unsqueeze(0).repeat(pred_proto.shape[0], 1, 1)], -1)  # b, n, 2c
        pred_proto = self.center_content_proj(pred_proto)
        
        feat = self.feat_proj(self.with_pos_embed(x, feat_pos))
        center = self.center_proj(self.with_pos_embed(pred_proto, center_pos))
        
        new_pred = self.get_pred(feat, center)
        return new_pred, pred_proto
    
    def get_class_diag(self, center):
        b, K, C = center.size()
        ones = torch.ones(K, dtype=torch.long)
        diag = torch.diag(ones) #(K, K)
        diag = diag.type(center.dtype)
        diag = diag.to(center.device).unsqueeze(0).repeat(b, 1, 1) #(b, k, k)

        return diag

    # center(b, k, C)
    def get_inter_center_relations(self, center):
        b, k, c = center.size()
        
        center = center / (torch.norm(center, 2, -1, True) + 1e-12)
        
        # scale = center.size(-1) ** -0.5
        center_p = center.permute(0, 2, 1).detach()
        attention = torch.matmul(center, center_p) * 15 # * scale
        
        attention = F.softmax(attention, dim=-1) #(b, k, k)
        
        diag = self.get_class_diag(center) #(b, K, K)
        return attention, diag
    
    def get_dis_loss(self, pred_proto, gt_proto, weight=10.0):
        pred_c2c_relation, diag = self.get_inter_center_relations(pred_proto)  #(b, k, k)
        gt_c2c_relation, diag = self.get_inter_center_relations(gt_proto)

        k = pred_c2c_relation.shape[-1]
        pred_other_relation = pred_c2c_relation * (1 - diag)  #(b, k, k)
        gt_other_relation = gt_c2c_relation * (1 - diag)  #(b, k, k)
        
        other_relation = pred_other_relation - gt_other_relation

        res_other_relation = torch.where(other_relation > 0, other_relation, torch.zeros_like(other_relation))

        loss = res_other_relation.sum(dim=-1) # (b)
        loss = loss.mean()

        return loss * weight
    
    def get_pos_dis_loss(self, center_pos, gt_center_pos, weight=0.4):
        b, k, c = center_pos.size()
        gt_center_pos = F.softmax(gt_center_pos / 1, -1)
        loss = torch.mul(-1 * F.log_softmax(center_pos, dim=-1), gt_center_pos)   # b, k, c
        loss = loss.sum(-1).mean()
        # print(loss)
        return loss * weight
    
    # attn (B, K, H, W) feat_pos (B, C, H, W), center_pos(k, c)
    def get_center_pos(self, attn, feat_pos):
        center_pos = self.center_pos.weight
        b, k, h, w = attn.size()
        c = feat_pos.shape[1]
        attn = attn.reshape(b, k, -1)
        feat_pos = feat_pos.reshape(b, c, -1).permute(0, 2, 1) #(b, hw, c)
        attn = F.softmax(attn, dim=-1) #(b, k, hw)
        center_pos = center_pos.unsqueeze(0).repeat(b, 1, 1)  #(b, k, c)
        center_pos = torch.cat([center_pos, torch.matmul(attn, feat_pos)], dim=-1) #(b, k, 2c)
        center_pos = self.center_pos_proj(center_pos)
        return center_pos #(b, k, c)
    
    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
        

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        y = gt_semantic_seg
        x, feat = self.forward(inputs)
        
        feat_pos = self.get_feat_pos(feat)
        center_pos = self.get_center_pos(x, feat_pos)

        pre_self_x = x.clone()
        x, pred_proto = self.pred_ssa(x=feat, pred=x, proto=self.conv_seg.weight.squeeze(), feat_pos=feat_pos, center_pos=center_pos)      
        ssa_pred, gt_proto, gt_center_pos = self.gt_ssa(x=feat, y=y, new_proto=self.conv_seg.weight.detach().data.squeeze(), proto=self.conv_seg.weight.squeeze(), feat_pos=feat_pos)   

        kl_loss = get_distill_loss(pred=x, soft=ssa_pred.detach(), target=y.squeeze(1))

        pre_self_x = F.interpolate(pre_self_x, size=y.shape[-2:], mode='bilinear', align_corners=True)
        pre_self_loss = self.criterion(pre_self_x, y.squeeze(1).long()) 
        ssa_pred = F.interpolate(ssa_pred, size=y.shape[-2:], mode='bilinear', align_corners=True)
        pre_loss = self.criterion(ssa_pred, y.squeeze(1).long()) 
        
        x = F.interpolate(x, size=y.shape[-2:], mode='bilinear', align_corners=True)
        outputs = {"pred_masks":x}
        dice_pred_loss = self.ssa_loss(outputs, y.squeeze(1))['loss_dice']

        outputs = {"pred_masks":ssa_pred}
        dice_gt_loss = self.ssa_loss(outputs, y.squeeze(1))['loss_dice']
        
        dis_loss = self.get_dis_loss(pred_proto, gt_proto.detach())
        pos_dis_loss = self.get_pos_dis_loss(center_pos, gt_center_pos.detach())

        losses = self.losses(x, y)

        losses['dice_pred'] = dice_pred_loss.detach().data
        losses['dice_gt'] = dice_gt_loss.detach().data
        
        losses['disLoss'] = dis_loss.detach().data
        losses['pos_disLoss'] = pos_dis_loss.detach().data
        
        losses['PreSelfLoss'] =  pre_self_loss.detach().data
        losses['PreLoss'] =  pre_loss.detach().data
        losses['KLLoss'] =  kl_loss.detach().data
        losses['MainLoss'] =  losses['loss_ce'].detach().data
        losses['loss_ce'] = losses['loss_ce'] + pre_self_loss + pre_loss + kl_loss + dice_pred_loss + dice_gt_loss + dis_loss + pos_dis_loss
        return losses      

    def forward_test(self, inputs, img_metas, test_cfg):
        x, feat = self.forward(inputs)
        feat_pos = self.get_feat_pos(feat)
        center_pos = self.get_center_pos(x, feat_pos)
        x, pred_proto = self.pred_ssa(x=feat, pred=x, proto=self.conv_seg.weight.squeeze(), feat_pos=feat_pos, center_pos=center_pos)      
        return x    

        
def get_bd(label, edge_pad=True, edge_size=3):
    label = label.cpu().numpy()
    edges = []
    for i in range(len(label)):
        one_label = label[i].astype(np.uint8)
        edge = cv2.Canny(one_label, 0.1, 0.2)
        edge = np.array(edge)
        kernel = np.ones((edge_size, edge_size), np.uint8)
        y_k_size = 6
        x_k_size = 6
        if edge_pad:
            edge = edge[y_k_size:-y_k_size, x_k_size:-x_k_size]
            edge = np.pad(edge, ((y_k_size,y_k_size),(x_k_size,x_k_size)), mode='constant')
        edge = (cv2.dilate(edge, kernel, iterations=1)>50)*1.0
        edges.append(edge)
        # img = cv2.cvtColor(edge, cv2.COLOR_BGR2Gray)
        test_label = np.stack([one_label, one_label, one_label], axis=-1)
    edges = np.array(edges)
    edges = torch.tensor(edges).cuda()
    return edges

def get_distill_loss(pred, soft, target, smoothness=0.5, eps=0):
    '''
    knowledge distillation loss
    '''
    b, c, h, w = soft.shape[:]
    soft.detach()
    target = F.interpolate(target.unsqueeze(1).float(), size=pred.shape[-2:], mode='nearest').squeeze(1).long()
    
    edges_mask = get_bd(target)
    
    onehot = target.view(-1, 1) # bhw, 1
    ignore_mask = (onehot == 255).float()
    onehot = onehot * (1 - ignore_mask) 
    onehot = torch.zeros(b*h*w, c).cuda().scatter_(1,onehot.long(),1)  # bhw, n
    onehot = onehot.contiguous().view(b, h, w, c).permute(0, 3, 1, 2)   # b, n, h, w
    sm_soft = F.softmax(soft / 1, 1)
    smoothed_label = smoothness * sm_soft + (1 - smoothness) * onehot
    if eps > 0: 
        smoothed_label = smoothed_label * (1 - eps) + (1 - smoothed_label) * eps / (smoothed_label.shape[1] - 1) 

    loss = torch.mul(-1 * F.log_softmax(pred, dim=1), smoothed_label)   # b, n, h, w
    
    sm_soft = F.softmax(soft / 1, 1)   # b, c, h, w    
    entropy_mask = -1 * (sm_soft * torch.log(sm_soft + 1e-12)).sum(1)
    loss = loss.sum(1) 

    ### for class-wise entropy estimation    
    unique_classes = list(target.unique())
    if 255 in unique_classes:
        unique_classes.remove(255)
        
    valid_mask = (target != 255).float()
    entropy_mask = entropy_mask * valid_mask
    
    loss_list = []
    weight_list = []
    for tmp_y in unique_classes:
        tmp_mask = (target == tmp_y).float()
        class_weight = 1
        
        tmp_entropy_mask = entropy_mask * tmp_mask * edges_mask
        tmp_loss1 = (loss * tmp_entropy_mask).sum() / (tmp_entropy_mask.sum() + 1e-12)
        
        tmp_entropy_mask = entropy_mask * tmp_mask * (1-edges_mask)
        tmp_loss2 = (loss * tmp_entropy_mask).sum() / (tmp_entropy_mask.sum() + 1e-12)
        
        tmp_loss = (tmp_loss1 + tmp_loss2) / 2.0
        
        loss_list.append(class_weight * tmp_loss)
        weight_list.append(class_weight)
    if len(weight_list) > 0:
        loss = sum(loss_list) / (sum(weight_list) + 1e-12)
    else:
        loss = torch.zeros(1).cuda().mean()
    return loss
