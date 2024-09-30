from mmcv.cnn import ConvModule

from mmseg.ops import resize
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import kaiming_init, constant_init
from timm.layers import DropPath
from timm.layers.helpers import to_2tuple
import math


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
        m[-1].inited = True
    else:
        constant_init(m, val=0)
        m.inited = True


class DPGHead(nn.Module):
    def __init__(self, in_ch, mid_ch, pool, fusions):
        super(DPGHead, self).__init__()
        assert pool in ['avg', 'att']
        assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
        assert len(fusions) > 0, 'at least one fusion should be used'
        self.inplanes = in_ch
        self.planes = mid_ch
        self.pool = pool
        self.fusions = fusions
        if 'att' in pool:
            self.conv_mask = nn.Conv2d(self.inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusions:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pool == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pool == 'att':
            #[N, D, C, 1]
            input_x = x
            input_x = input_x.view(batch, channel, height*width) # [N, D, C]
            input_x = input_x.unsqueeze(1) # [N, 1, D, C]

            context_mask = self.conv_mask(x) # [N, 1, C, 1]
            context_mask = context_mask.view(batch, 1, height*width) # [N, 1, C]
            context_mask = self.softmax(context_mask) # [N, 1, C]
            context_mask = context_mask.unsqueeze(3) # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)# [N, 1, D, 1]
            context = context.view(batch, channel, 1, 1) # [N, D, 1, 1]
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x, y):
        # [N, C, 1, 1]
        context = self.spatial_pool(y)

        if self.channel_mul_conv is not None:
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))# [N, D, 1, 1]
            out = x * channel_mul_term # [N, D, H, W]
        else:
            out = x
        if self.channel_add_conv is not None:
            channel_add_term = self.channel_add_conv(context)# [N, D, 1, 1]
            out = out + channel_add_term

        return out

class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    copied from timm: https://github.com/huggingface/pytorch-image-models/blob/v0.6.11/timm/models/layers/mlp.py
    """
    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU,
            norm_layer=None, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class RCA(nn.Module):
    def __init__(self, inp,  kernel_size=1, ratio=1, band_kernel_size=11,dw_size=(1,1), padding=(0,0), stride=1, square_kernel_size=2, relu=True):
        super(RCA, self).__init__()
        self.dwconv_hw = nn.Conv2d(inp, inp, square_kernel_size, padding=square_kernel_size//2, groups=inp)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        gc=inp//ratio
        self.excite = nn.Sequential(
                nn.Conv2d(inp, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size//2), groups=gc),
                nn.BatchNorm2d(gc),
                nn.ReLU(inplace=True),
                nn.Conv2d(gc, inp, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size//2, 0), groups=gc),
                nn.Sigmoid()
            )
    
    def sge(self, x):
        #[N, D, C, 1]
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)
        x_gather = x_h + x_w #.repeat(1,1,1,x_w.shape[-1])
        ge = self.excite(x_gather) # [N, 1, C, 1]
        
        return ge

    def forward(self, x):
        loc=self.dwconv_hw(x)
        att=self.sge(x)
        out = att*loc
        
        return out

class RCM(nn.Module):
    """ MetaNeXtBlock Block
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
            self,
            dim,
            token_mixer=RCA,
            norm_layer=nn.BatchNorm2d,
            mlp_layer=ConvMlp,
            mlp_ratio=2,
            act_layer=nn.GELU,
            ls_init_value=1e-6,
            drop_path=0.,
            dw_size=11,
            square_kernel_size=3,
            ratio=1,
    ):
        super().__init__()
        self.token_mixer = token_mixer(dim, band_kernel_size=dw_size, square_kernel_size=square_kernel_size, ratio=ratio)
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.token_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class PyramidPoolAgg(nn.Module):
    def __init__(self, stride=2):
        super().__init__()
        self.stride = stride

    def forward(self, inputs):
        B, C, H, W = inputs[-1].shape
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        return torch.cat([nn.functional.adaptive_avg_pool2d(inp, (H, W)) for inp in inputs], dim=1)

class FuseBlockMulti(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int = 1,
        norm_cfg=dict(type='BN', requires_grad=True),
        activations = None,
    ) -> None:
        super(FuseBlockMulti, self).__init__()
        self.stride = stride
        self.norm_cfg = norm_cfg
        assert stride in [1, 2]

        if activations is None:
            activations = nn.ReLU

        self.fuse1 = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.fuse2 = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.act = h_sigmoid()

    def forward(self, x_l, x_h):
        B, C, H, W = x_l.shape
        inp = self.fuse1(x_l)
        sig_act = self.fuse2(x_h)
        sig_act = F.interpolate(self.act(sig_act), size=(H, W), mode='bilinear', align_corners=False)
        out = inp * sig_act
        return out

class NextLayer(nn.Module):
    def __init__(self, block_num, embedding_dim, dw_size, module=RCM, mlp_ratio=2, token_mixer=RCA, square_kernel_size=3):#
        super().__init__()
        self.block_num = block_num

        self.transformer_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.transformer_blocks.append(module(embedding_dim, token_mixer=token_mixer, dw_size=dw_size, mlp_ratio=mlp_ratio, square_kernel_size=square_kernel_size))

    def forward(self, x):
        # token * N 
        for i in range(self.block_num):
            x = self.transformer_blocks[i](x)
        return x

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

@HEADS.register_module()
class CGRSeg(BaseDecodeHead):
    def __init__(self, is_dw=False, next_repeat=4, mr=2, dw_size=7, neck_size=3, square_kernel_size=1, module='RCA', ratio=1, **kwargs):
        super(CGRSeg, self).__init__(input_transform='multiple_select', **kwargs)
        embedding_dim = self.channels

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            kernel_size=1,
            stride=1,
            groups=embedding_dim if is_dw else 1,
            norm_cfg=self.norm_cfg, 
            act_cfg=self.act_cfg
        )
        self.ppa=PyramidPoolAgg(stride=2)
        norm_cfg = dict(type='SyncBN', requires_grad=True)
        act_layer=nn.ReLU6
        module_dict={
            'RCA':RCA,
        }
        self.trans=NextLayer(next_repeat, sum(self.in_channels), dw_size=neck_size, mlp_ratio=mr, token_mixer=module_dict[module], square_kernel_size=square_kernel_size)
        self.SIM = nn.ModuleList() 
        self.meta = nn.ModuleList() 
        for i in range(len(self.in_channels)):
            self.SIM.append(FuseBlockMulti(self.in_channels[i], self.channels, norm_cfg=norm_cfg, activations=act_layer))
            self.meta.append(RCM(self.in_channels[i],token_mixer=module_dict[module], dw_size=dw_size, mlp_ratio=mr, square_kernel_size=square_kernel_size, ratio=ratio))
        self.conv=nn.ModuleList()
        for i in range(len(self.in_channels)-1):
            self.conv.append(nn.Conv2d(self.channels, self.in_channels[i], 1))

        self.spatial_gather_module=SpatialGatherModule(1)
        self.lgc=DPGHead(embedding_dim, embedding_dim, pool='att', fusions=['channel_mul'])

    def forward(self, inputs):
        xx = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        out=self.ppa(xx)
        out = self.trans(out)
        f_cat = out.split(self.in_channels, dim=1)
        results = []
        for i in range(len(self.in_channels)-1,-1,-1):
            if i==len(self.in_channels)-1:
                local_tokens = xx[i]
            else:
                local_tokens = xx[i]+ self.conv[i](F.interpolate(results[-1], size=xx[i].shape[2:], mode='bilinear', align_corners=False))
            global_semantics = f_cat[i]
            local_tokens=self.meta[i](local_tokens)
            flag = self.SIM[i](local_tokens, global_semantics)
            results.append(flag)
        x = results[-1]
        _c = self.linear_fuse(x)
        prev_output = self.cls_seg(_c)

        context = self.spatial_gather_module(x, prev_output) #8*128*150*1
        object_context = self.lgc(x, context)+x #8*128*8*8
        output = self.cls_seg(object_context)

        return output