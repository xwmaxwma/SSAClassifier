# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from mmseg.models.builder import HEADS
from mmseg.models.utils import SelfAttentionBlock as _SelfAttentionBlock
from mmseg.models.decode_heads.cascade_decode_head import BaseCascadeDecodeHead

from models.losses import SSALoss
import cv2
import numpy as np


class SpatialGatherModule(nn.Module):
    """Aggregate the context features according to the initial predicted
    probability distribution.

    Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, scale):
        super(SpatialGatherModule, self).__init__()
        self.scale = scale

    def forward(self, feats, probs):
        """Forward function."""
        batch_size, num_classes, height, width = probs.size()
        channels = feats.size(1)
        probs = probs.view(batch_size, num_classes, -1)
        feats = feats.view(batch_size, channels, -1)
        # [batch_size, height*width, num_classes]
        feats = feats.permute(0, 2, 1)
        # [batch_size, channels, height*width]
        probs = F.softmax(self.scale * probs, dim=2)
        # [batch_size, channels, num_classes]
        ocr_context = torch.matmul(probs, feats)
        ocr_context = ocr_context.permute(0, 2, 1).contiguous().unsqueeze(3)
        return ocr_context


class ObjectAttentionBlock(_SelfAttentionBlock):
    """Make a OCR used SelfAttentionBlock."""

    def __init__(self, in_channels, channels, scale, conv_cfg, norm_cfg,
                 act_cfg):
        if scale > 1:
            query_downsample = nn.MaxPool2d(kernel_size=scale)
        else:
            query_downsample = None
        super(ObjectAttentionBlock, self).__init__(
            key_in_channels=in_channels,
            query_in_channels=in_channels,
            channels=channels,
            out_channels=in_channels,
            share_key_query=False,
            query_downsample=query_downsample,
            key_downsample=None,
            key_query_num_convs=2,
            key_query_norm=True,
            value_out_num_convs=1,
            value_out_norm=True,
            matmul_norm=True,
            with_out=True,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.bottleneck = ConvModule(
            in_channels * 2,
            in_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, query_feats, key_feats):
        """Forward function."""
        context = super(ObjectAttentionBlock,
                        self).forward(query_feats, key_feats)
        output = self.bottleneck(torch.cat([context, query_feats], dim=1))
        if self.query_downsample is not None:
            output = resize(query_feats)

        return output


@HEADS.register_module()
class OCRHead_ssa(BaseCascadeDecodeHead):
    """Object-Contextual Representations for Semantic Segmentation.

    This head is the implementation of `OCRNet
    <https://arxiv.org/abs/1909.11065>`_.

    Args:
        ocr_channels (int): The intermediate channels of OCR block.
        scale (int): The scale of probability map in SpatialGatherModule in
            Default: 1.
    """

    def __init__(self, ocr_channels, scale=1, **kwargs):
        super(OCRHead_ssa, self).__init__(**kwargs)
        self.ocr_channels = ocr_channels
        self.scale = scale
        self.object_context_block = ObjectAttentionBlock(
            self.channels,
            self.ocr_channels,
            self.scale,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.spatial_gather_module = SpatialGatherModule(self.scale)

        self.bottleneck = ConvModule(
            self.in_channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        
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

    def forward(self, inputs, prev_output):
        """Forward function."""
        x = self._transform_inputs(inputs)
        feats = self.bottleneck(x)
        context = self.spatial_gather_module(feats, prev_output)
        object_context = self.object_context_block(feats, context)
        output = self.cls_seg(object_context)

        return output, object_context
    
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
        

    def forward_train(self, inputs, prev_output, img_metas, gt_semantic_seg,
                      train_cfg):
        y = gt_semantic_seg
        x, feat = self.forward(inputs, prev_output)
        
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

    def forward_test(self, inputs, prev_output, img_metas, test_cfg):
        x, feat = self.forward(inputs, prev_output)
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

