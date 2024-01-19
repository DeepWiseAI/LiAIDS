#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import os.path as osp
import sys
import json
import time 
import numpy as np
import pandas as pd
from scipy.ndimage import label
import SimpleITK as sitk

import torch
from d0_data_parser import LiverPatient, class_dict, segment_dict, phase_dict, get_cls_dict
from tumor_mark import tumor_mark, merge_box



class AI_LiverDetect():
    def init(self, gpu_id):
        self.gpu_id = gpu_id
        conf = json.loads(open("func.conf").read())
        self.device = torch.device("cuda:{}".format(self.gpu_id) if self.gpu_id >= 0 else "cpu")
        self.conf = conf

        self.patient = None
        self.debug = self.conf['B_debug']
        return os.getpid()

    #  Init Model
    def prepare(self):
        from get_liver_tumor_seg import LiverTumorSegmentation  
        from get_nodule_det import NoduleDet
        from get_tumor_cls import ClsModel
        torch.cuda.set_device(int(self.gpu_id))
        
        liver_seg_model_folder = self.conf["liver_seg_model"]["model_folder"]
        liver_seg_folds = self.conf["liver_seg_model"]["folds"]
        checkpoint_name = self.conf["liver_seg_model"]["checkpoint_name"]
        self.lits = LiverTumorSegmentation(liver_seg_model_folder, liver_seg_folds, checkpoint_name=checkpoint_name)
        self.tumor_detect = NoduleDet(self.conf["liver_dete_model"], self.gpu_id)
        self.tumor_cls = ClsModel(self.conf["tumor_cls_model"], self.gpu_id)

        return True


    @torch.no_grad()
    def main_course(self, input_dir):
        '''
        input_dir: contains CT series organized in a sorting sequence of non-phase, A-phase, V-phase, and ...
        '''
        
        # Decode input_dir if not a string
        if not isinstance(input_dir, str):
            input_dir = input_dir.decode("utf-8")
        
        # Process INPUT_DIR
        hou_input_dirs = input_dir.split(",")

        # Check for missing phases in hou_input_dirs
        if '?' in hou_input_dirs[:3]:
            err_msg = 'INPUT_DIR parse failed: lack phase'
            return None, err_msg
        hou_input_dirs = [i for i in hou_input_dirs if i != '?']
        
        # Initialize patient and preprocess data
        self.patient = LiverPatient()
        self.patient.preprocess(hou_input_dirs)

        # Segmentation
        mask_dict = self.lits.get_liver_tumor_seg(self.patient.sub_dir_img)
        tmp_sub_dirs = mask_dict.keys()
        for sub_dir in tmp_sub_dirs:
            if sub_dir not in self.patient.sub_dirs:
                mask_dict.pop(sub_dir)
        torch.cuda.empty_cache()


        # Detection
        dete_dict, box_dict = self.tumor_detect.inference(self.patient.sub_dir_img, mask_dict)
        torch.cuda.empty_cache()
        
        # Data processing
        cls_dict = get_cls_dict() 
        default_keys = ["sub_dir", "liver_seg_begin", "liver_seg_end", "liver_seg_begin_slice", "liver_seg_end_slice", "phase", "orgin_x", "orgin_y", "z_min_location", 
        "z_max_location", "spacing_x", "spacing_y", "slice_thickness", "pid", "liver_seg_xmin", "liver_seg_xmax", "liver_seg_ymin", "liver_seg_ymax"] # List of default keys

        # Iterate through mask_dict to extract information in default_keys and cls_dict
        for sub_dir in mask_dict:
            # Extract information and populate cls_dict
            image = self.patient.sub_dir_img[sub_dir]
            tmp_mask = mask_dict[sub_dir].copy()
            cls_dict = get_info_for_defaule_keys()
            
        # Create DataFrame from cls_dict
        df = pd.DataFrame(cls_dict)
        
        # Additional processing steps
        # Using Intersection over Union (IOU) and Intersection over Minimum (IOM) metrics to 
        # determine whether lesions on different phases belong to the same lesion and assigning lesion identifiers.
        df = tumor_mark(df) # tumor register

        # Clinical data integration
        # Retrieve clinical information corresponding to the data, 
        # including 'Hepatitis B', 'Hepatitis C', 'Cirrhosis', 'History of Extrahepatic Tumors', 'Gender', and 'Age'
        clinical_info = get_clinica_info()

        # Nodule classification
        nodules_cls_result, nodule_dict = self.tumor_cls.test(df, self.patient.sub_dir_img, mask_dict, box_dict, clinical_info)
        
        # Final processing steps
        # Update DataFrame with classification results
        

        # Create nodes
        all_sub_dir_instance = self.patient.sub_dir_instance.copy() 
        self.patient.create_nodes(df, mask_dict, all_sub_dir_instance, box_dict)
        return json.dumps(self.patient.json), "OK"

