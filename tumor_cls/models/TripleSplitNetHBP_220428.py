# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict


class Densenet36_SE_keepz_featmap(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        n_classes (int) - number of classification classes
    """

    def __init__(self, **kwargs):
        super(Densenet36_SE_keepz_featmap, self).__init__()

        # Three feature blocks with _DenseBlock, _Transition and SELayer.  
        # The "_DenseBlock" and "_Transition" components are integral parts of the DenseNet architecture.
        # For detailed implementation, you can refer to  https://github.com/xmuyzz/3D-CNN-PyTorch/blob/master/models/DenseNet.py
        # SELayer can be found in https://github.com/moskomule/senet.pytorch/tree/master/senet 
        self.features_block1 = nn.Sequential(_DenseBlock, _Transition, SELayer)
        self.features_block1 = nn.Sequential(_DenseBlock, _Transition, SELayer)
        self.features_block1 = nn.Sequential(_DenseBlock, _Transition, SELayer)


        # Define the final convolutional block
        self.final_conv = nn.Sequential(OrderedDict([
            ('conv6', nn.Conv3dd),
            ('norm7', nn.BatchNorm3dd),
            ('relu8', nn.ReLUd)
        ]))

        # Define average pooling layers 
        self.average_pool_2 = nn.AvgPool3d()
        self.average_pool_4 = nn.AvgPool3d()
        self.average_pool_8 = nn.AvgPool3d()

        # Define self-attention layer
        # 3D implementation of self-attention in the "Self-Attention Generative Adversarial Networks" <https://arxiv.org/abs/1805.08318>
        # https://github.com/voletiv/self-attention-GAN-pytorch/blob/master/sagan_models.py
        self.sa1 = Self_Attn()

    def forward(self, x):
        out = self.features_block1(x)
        out_b1 = out

        out = self.features_block2(out)
        out_b2 = out
        out = self.features_block3(out)
        out_b3 = out
        out = self.features_block4(out)

        out = F.relu(out)
        out_b4 = out
        out_b1 = self.average_pool_8(out_b1)
        out_b2 = self.average_pool_4(out_b2)
        out_b3 = self.average_pool_2(out_b3)

        out = torch.cat((out_b1, out_b2, out_b3, out_b4), 1)
        out = self.final_conv(out)

        return out



def Densenet36_fgpn_se_keepz_featmap(**kwargs):
    model = Densenet36_SE_keepz_featmap(**kwargs) 
    return model



class TripleSplitNetHBP_220428(nn.Module):
    def __build_model(self, nchannel):
        return Densenet36_fgpn_se_keepz_featmap(n_input_channels = nchannel)

    def __init__(self, nchannel, num_classes=1):
        super(TripleSplitNetHBP_220428, self).__init__()
        self.embedding_net1 = self.__build_model(nchannel)
        self.embedding_net2 = self.__build_model(nchannel)
        self.embedding_net3 = self.__build_model(nchannel)

        # A reimplementation of Hierarchical Bilinear Pooling for Fine-Grained Visual Recognition
        # https://github.com/Ylexx/Hierarchical-Bilinear-Pooling_Resnet_Pytorch/blob/master/hbp_model.py
        self.hbp = HBP()

    def forward(self, input1, input2, input3):

        # Extract features from each input using the three embedding networks
        # Defined the weight of each embedding net
        feature1 = self.embedding_net1(input1) * weight[0] 
        feature2 = self.embedding_net2(input2) * weight[1]
        feature3 = self.embedding_net3(input3) * weight[2]
        output = self.hbp(feature1, feature2, feature3)
        return output  




