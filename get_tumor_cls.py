# -*- coding:utf-8 -*-

import argparse
import os
import time
import os.path as op
import sys

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import zoom
from skimage.transform import resize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from tumor_cls import data_crops 
from tumor_cls import models
from tumor_cls.models.merge_model import Merge, Clinical_cls
from tumor_cls.configs.config_220428 import Config7Type



class ClsModel:
    def __init__(self, conf, gpu_id):
        # Initialize the class attributes
        self.gpu_id = gpu_id
        self.para_config = Config7Type()
        
        # Initialize data crop and phase list
        data_crop_name = self.para_config.data_crop_name
        self.data_crop = data_crops.init_datacrop(data_crop_name)
        self.phase_list = self.para_config.phase_list
        
        # Initialize model and fusion model with specified configurations
        model_name = self.para_config.model_name 
        weight_path = conf["weight_path"]
        fusion_weight_path = conf["fusion_weight_path"]

        # Initialize and load weights for the main model
        model = models.init_model(model_name, nchannel=self.para_config.nchannel, num_classes=len(self.para_config.sub_class_dict))
        fusion_model = Clinical_cls()
        self.model, _, = load_weights(model, weight_path)

        # Initialize and load weights for the fusion model
        self.fusion_model, _, = load_weights(fusion_model, fusion_weight_path)
        self.model.eval()
        self.fusion_model.eval()


    def test(self, df, sub_dir_img, sub_dir_livermask, box_dict, clinical_info):
        """
        Args:
        df (pandas) - The basic information of lesions in the data, including spacing, 
        the initial position of the liver, physical location information, etc
        sub_dir_img (dict) - Saved data for all phases
        sub_dir_livermask (dict)) - Stored segmentation results of the liver for all data
        box_dict (dict) - Detection results
        clinical_info - Clinical information
        """

        clinical_info = torch.from_numpy(clinical_info)
        clinical_info = clinical_info.unsqueeze(0).float().cuda(self.gpu_id)

        # Perform data cropping
        pid_nodule_dict, box_dict = self.data_crop.data_crop_func(df, sub_dir_img, sub_dir_livermask, box_dict)
    
        target_shape = self.para_config.final_size_shape
        final_crop_size_shape=self.para_config.final_crop_size_shape

        result = {}
        self.model.cuda(self.gpu_id)
        self.fusion_model.cuda(self.gpu_id)

        feature = []
        nodule_list = []

        # The predictions for various lesions can be summarized as follows:
        """
        # Loop through nodules
            # Extract coordinates from pid_nodule_dict
            # Loop through phases
                # Load phase image
                # Resize and preprocess images
            # End phases loop
            # Concatenate all phase images along axis 0
            # Center crop images
            # Process features using self.model, where the input consists of nodule patches with arterial, venous, and contrast components
            # generate predictions by self.fusion_model, where the input consists feature maps from self.model and clinical information
            # save to nodule_list
        # End nodules loop
        # result = F.softmax(nodule_list)
        """
   
        return result, pid_nodule_dict
    

   
