#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


class Clinical_cls(nn.Module):
    def __init__(self, num_classes=7):
        super(Clinical_cls, self).__init__()
        

        #  Define the simple classification network
        #  The combination of a fully connected layer and ReLU activation
        self.simpleCls = nn.Sequential()

    def forward(self, x, clinic_info):
        # Concatenate input features with clinical information
        x = torch.cat((x, clinic_info), 1)
        out = self.simpleCls(x)
        return out