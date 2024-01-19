# ------------------------------------------
# CSWin Transformer
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# written By Xiaoyi Dong
# ------------------------------------------

"""
Based on "CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows", <https://arxiv.org/abs/2107.00652>
The original code can be found at https://github.com/microsoft/CSWin-Transformer/blob/main/models/cswin.py

We have made modifications to the original code, particularly in functions like LePEAttention and Merge_Block, to adapt it for 3D input
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from timm.models.layers import DropPath, trunc_normal_
from mmcv.runner import load_checkpoint
from mmdet2_15_0.utils import get_root_logger
from ..builder import BACKBONES


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x




class LePEAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        """Not supported now, since we have cls_tokens now.....
        """
        super().__init__()




class CSWinBlock(nn.Module):

    def __init__(self, dim, patches_resolution, num_heads,
                 split_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False):
        super().__init__()
        


class Merge_Block(nn.Module):
    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
        super().__init__()

        # 3D convolution layer with kernel size 3, stride 2, and padding 1
        self.conv
        # Normalization layer 
        self.norm 

    def forward(self, x, H, W, D):
        B, new_HW, C = x.shape
        # Reshape the input tensor
        # Apply 3D convolution
        # Reshape the tensor
        # # Apply normalization
        return x, H, W, D



@BACKBONES.register_module()
class CSWin(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, **args):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        heads=num_heads
        self.use_chk = use_chk

        # 3D conv
        self.stage1_3d 
        self.stage1_conv_embed = nn.Sequential(
            nn.Conv2d,
            nn.LayerNorm
        )

        self.norm1 = nn.LayerNorm(embed_dim)

        curr_dim = embed_dim
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))]  # stochastic depth decay rule

        self.stage1 = nn.ModuleList([
            CSWinBlock()
            for i in range(depth[0])])

        self.merge1 = Merge_Block(curr_dim, curr_dim*(heads[1]//heads[0]))
        curr_dim = curr_dim*(heads[1]//heads[0])
        self.norm2 = nn.LayerNorm(curr_dim)
        self.stage2 = nn.ModuleList(
            [CSWinBlock()
            for i in range(depth[1])])
        
        self.merge2 = Merge_Block()
        curr_dim = curr_dim*(heads[2]//heads[1])
        self.norm3 = nn.LayerNorm(curr_dim)
        temp_stage3 = []
        temp_stage3.extend(
            [CSWinBlock()
            for i in range(depth[2])])

        self.stage3 = nn.ModuleList(temp_stage3)
        
        self.merge3 = Merge_Block()
        curr_dim = curr_dim*(heads[3]//heads[2])
        self.stage4 = nn.ModuleList(
            [CSWinBlock()
            for i in range(depth[-1])])
       
        self.norm4 = norm_layer(curr_dim)


    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')


    def center_cropping(self, x):
        """Center crop a N*C*D*H*W 3D feature map to N*C*H*W"""
        
        return center_x


    def save_out(self, x, norm, H, W, D):
        x = norm(x)
        B, N, C = x.shape
        x = x.view(B, H, W, C)
        x = x.view(B//D, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        x = self.center_cropping(x)
        return x

    def forward_features(self, x):
        B = x.shape[0]
        # Stage 1: 3D convolution
        x = self.stage1_3d(x)

        # Reshape and apply initial convolution
        # Process the blocks in stage 1
        # Save the output for stage 1
        # Process stages 2, 3, and 4
        # Save the output for the current stage


        return tuple(out)


    def forward(self, x):
        x = self.forward_features(x)
        return x



def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict

