import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from torch.nn.functional import interpolate


norm_func = nn.InstanceNorm3d
norm_func2d = nn.InstanceNorm2d
act_func = nn.CELU

import torch.nn.functional
from torch.nn.functional import fold, unfold
from math import *



class Swish(nn.Module):
    def __init__(self, inplace):
        super(Swish, self).__init__() 
    def forward(self, x):
        return x*x.sigmoid()

class PassNorm(nn.Module):
    def __init__(self, inplace):
        super(PassNorm, self).__init__() 
    def forward(self, x):
        return x

act_func = Swish
# norm_func = PassNorm


class ResConv2dNormAct(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, groups):
        super(ResConv2dNormAct, self).__init__()
        self.falg = (inchannel==outchannel)

        # Convolutional block with 1x1 convolution
        self.conv0 = nn.Sequential()

        # Normalization and activation block
        self.norm_act = nn.Sequential()

        # Convolutional block with grouped convolutions (kernel_size x kernel_size and kernel_size-2 x kernel_size-2)
        self.conv1_5 = nn.Sequential()
        self.conv1_3 = nn.Sequential()

        # Final convolutional block with normalization, activation, and scSE, 
        # scSE Attention Unit is located at https://gitcode.net/mirrors/shanglianlm0525/pytorch-networks/-/blob/master/Attention/SEvariants.py; 
        self.conv2d = nn.Sequential()

    def forward(self, x):
        x = self.conv0(x)
        norm = self.norm_act(x).chunk(2, 1)
        x = x + torch.cat([self.conv1_3(norm[0]), self.conv1_5(norm[1])], dim=1)
        return self.conv2(x)



class ResConv3dNormAct(nn.Module):
    def __init__(self, inchannel, outchannel, att=False, num_heads=2):
        super(ResConv3dNormAct, self).__init__()
        self.num_heads = num_heads

        # Convolutional block, convolutional with kernel 3, normalization and activation block
        self.conv_0 = nn.Sequential()

        # Attention mechanism
        if att:
            self.conv1 = nn.ModuleList()
            for i in range(num_heads):
                # SwinBlock: CSwin Transformer can be found at https://github.com/microsoft/CSWin-Transformer
                self.conv1.append(SwinBlock())
        else: 
            # Convolutional block, convolutional with kernel 3, normalization and activation block
            self.conv1 = nn.Sequential()
        self.att = att
    

    def forward(self, x):
        out = self.conv_0(x)
        if self.att:
            out = out.chunk(self.num_heads, 1)
            outs = []
            for index in range(self.num_heads):
                outs.append(self.conv1[index](out[index]))
            out = torch.cat(outs, dim=1)
        else: 
            out = out + self.conv1(out)
        return out


class Conv2dUnit(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size=3):
        super(Conv2dUnit, self).__init__()

        # Define the convolutional unit with a sequence of operations
        self.conv = nn.Sequential()

    def forward(self, x):
        return self.conv(x)


class DownConv(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride, if_att=True, num_heads=2):
        super(DownConv, self).__init__()

        # Define the down convolutional block with max pooling followed by ResConv3dNormAct
        self.conv = nn.Sequential()

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    def __init__(self, inchannel, outchannel, stride, if_att=False, num_heads=2):
        super(UpConv, self).__init__()

        # Define the up convolutional block with upsampling, convolution, normalization, and activation
        self.up_conv = nn.Sequential()

    def forward(self, x):
        return self.up_conv(x)



class LiverSegNet(nn.Module):

    def __init__(self,
                 inchannel=1,
                 num_seg_classes=2):
        super(LiverSegNet, self).__init__()
        

        # Define the preprocessing 3D convolutional block
        self.pre_conv = nn.Sequential(norm_func,
            nn.Conv3d, # with inchannel, base_channel, kernel=3
            norm_func, nn.GELU(),
        )

        # Define the encoder blocks
        # 2D Convolutional block with AvgPool and ResConv2dNormAct 
        self.conv0 = nn.Sequential(nn.AvgPool2d, ResConv2dNormAct)
        self.conv1 = nn.Sequential(nn.AvgPool2d, ResConv2dNormAct) 

        # 2D Convolutional block with MaxPool and two ResConv2dNormAct blocks
        self.conv2 = nn.Sequential(nn.MaxPool2d, ResConv2dNormAct, ResConv2dNormAct) 
        self.conv3 = nn.Sequential(nn.MaxPool2d, ResConv2dNormAct, ResConv2dNormAct) 
        self.conv4 = nn.Sequential(nn.MaxPool2d, ResConv2dNormAct, ResConv2dNormAct) #16x16

        # Convolutional block with DownConv and NonLocal, 
        # NonLocal can be found in https://github.com/tea1528/Non-Local-NN-Pytorch/blob/master/models/non_local.py
        self.conv5 = nn.Sequential(DownConv, NonLocal)
        self.conv6 = nn.Sequential(DownConv, NonLocal)

        # Define the decoder blocks
        self.up6 = nn.Sequential(UpConv)

        # Up Convolutional block P5 and Up5
        self.p5 = nn.Sequential(ResConv3dNormAct,NonLocal)
        self.up5 = UpConv()

        # Convolutional block with two ResConv2dNormAct blocks 
        # Up Convolutional block with Upsample and Conv2dUnit
        self.p4 = nn.Sequential(ResConv2dNormAct,ResConv2dNormAct)
        self.up4 = nn.Sequential(nn.Upsample, Conv2dUnit)

        self.p3 = nn.Sequential(ResConv2dNormAct, ResConv2dNormAct)
        self.up3 = nn.Sequential(nn.Upsample, Conv2dUnit)

        self.p2 = nn.Sequential(ResConv2dNormAct, ResConv2dNormAct)
        self.up2 = nn.Sequential(nn.Upsample, Conv2dUnit)
        
        self.p1 = nn.Sequential(ResConv2dNormAct, ResConv2dNormAct)
        self.up1 = nn.Sequential(nn.Upsample, Conv2dUnit)

        self.p0 = nn.Sequential(ResConv2dNormAct,ResConv2dNormAct)

        
        # Define the final convolutional layers for segmentation
        # with  convolution, normalization, and activation
        self.conv_last = nn.Sequential()
        self.conv_seg_2 = nn.Sequential()
        self.conv_seg_3 = nn.Sequential()
        self.conv_seg_4 = nn.Sequential()


    def upsample_depth(self, x, n):
        # Upsample along the depth dimension
        return x


    def forward(self, x):
        # Apply the preprocessing convolutional block
        x = self.pre_conv(x)
        n, C, depth, H, W = x.shape

        # Reshape the input tensor
        x = x.transpose(1,2).contiguous().view(n*depth, C, H, W) #32x512x512

        # Encoder blocks
        conv0 = self.conv0(x)
        _conv0 = (conv0[::2] + conv0[1::2])/2
        depth = depth//2

        conv1 = self.conv1(_conv0) #16x128x128
        _conv1 = (conv1[::2] + conv1[1::2])/2
        depth = depth//2

        conv2 = self.conv2(_conv1) #8x64x64
        _conv2 = (conv2[::2] + conv2[1::2])/2
        depth = depth//2

        conv3 = self.conv3(_conv2) #4x32x32
        conv4 = self.conv4(conv3) #4x16x16

        N, C, H, W = conv4.shape
        _conv4 = conv4.view(N//depth, depth, C, H, W).transpose(1,2).contiguous()

        conv5 = self.conv5(_conv4)
        conv6 = self.conv6(conv5)

        # Decoder blocks
        p6 = self.up6(conv6)

        p5 = self.up5(self.p5(torch.cat([conv5, p6], dim=1)))
        p5 = p5.transpose(1,2).contiguous().view(N, C, H, W)
        p5 = self.p4(torch.cat([p5, conv4], dim=1))
        p4 = self.up4(p5)
        p4 = self.p3(torch.cat([p4, conv3], dim=1))
        p3 = self.upsample_depth(self.up3(p4), n)
        depth = depth*2

        p3 = self.p2(torch.cat([p3, conv2], dim=1))
        p2 = self.upsample_depth(self.up2(p3), n)
        depth = depth*2
        p2 = self.p1(torch.cat([p2, conv1], dim=1))
        p1 = self.upsample_depth(self.up1(p2), n)
        depth = depth*2

        p1 = self.p0(torch.cat([p1, conv0], dim=1))

        # Segmentation blocks
        C, H, W = p4.shape[1:]
        seg_4 = self.conv_seg_4(p4.unsqueeze(0).view(n, -1, C, H, W).transpose(1,2).contiguous())
        C, H, W = p3.shape[1:]
        seg_3 = self.conv_seg_3(p3.unsqueeze(0).view(n, -1, C, H, W).transpose(1,2).contiguous())
        C, H, W = p2.shape[1:]
        seg_2 = self.conv_seg_2(p2.unsqueeze(0).view(n, -1, C, H, W).transpose(1,2).contiguous())
        C, H, W = p1.shape[1:]
        seg_1 = self.conv_last(p1.unsqueeze(0).view(n, -1, C, H, W).transpose(1,2).contiguous())
        out_last = [seg_4, seg_3, seg_2, seg_1]

        return out_last


def vesselnet(**kwargs):
    model = VesselNet(**kwargs)
    return model