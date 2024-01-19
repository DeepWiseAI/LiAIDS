import os
import os.path as osp
import glob
import pdb
import time
import numpy as np
import pandas as pd

import mmcv
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmdet2_15_0.datasets import build_dataloader, build_dataset, build_ct_dataset
from mmdet2_15_0.models import build_detector
from mmcv.runner import (load_checkpoint,                                                                                                       
                         wrap_fp16_model)

import torch
import torch.distributed as dist
from nodule_det.ct_utils.eval_classes import TOP_CLASS
from nodule_det.ct_utils.convert_utils import convert_pred


class NoduleDet():
    def __init__(self, conf, gpu_id):
        self.conf = conf

        self.model_config = self.conf['config_path']
        self.model_pth = self.conf['cpt_path']
        score_thresh = self.conf['score_thresh']
        self.score_thresh_list = [0] + [score_thresh] * len(TOP_CLASS)
        self.minimum_slice_num = self.conf['minimum_slice_num']
        self.max_slices_stride = self.conf['max_slices_stride']
        self.iom_thresh = self.conf['iom_thresh']
        self.post_thresh = 0.2
        if 'post_thresh' in self.conf:
            self.post_thresh = self.conf['post_thresh']
        self.gpu_id = gpu_id
        self._init_model()

    def _init_model(self):
        self.cfg = mmcv.Config.fromfile(self.model_config)
        self.cfg.model.pretrained = None
        self.cfg.data.test_ct_all.test_mode = True
        self.model = build_detector(self.cfg.model, train_cfg=None, test_cfg=self.cfg.get('test_cfg'))
        fp16_cfg = self.cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(self.model)
 
        load_checkpoint(self.model, self.model_pth, map_location='cpu')
        self.model = MMDataParallel(self.model, device_ids = [self.gpu_id])

    def _infer_2d(self, sub_dir_list, mask_dict):
        self.model.eval()
        all_results_dicts = {}

        # Loop through sub_dir_list
        for sub_dir in sub_dir_list:
            # Set configuration parameters for the current sub-directory
            self.cfg.data.test_ct_all.sub_dir = sub_dir_list[sub_dir]
            self.cfg.data.test_ct_all.mask_tensor = mask_dict[sub_dir]

            # Build dataset and data loader
            dataset = build_ct_dataset(self.cfg.data.test_ct_all)
            data_loader = build_dataloader(
                dataset, 
                samples_per_gpu = 4, 
                workers_per_gpu = 1, 
                dist = False, 
                shuffle = False)

            results_dict = {}
            # Iterate through batches in the data loader
            for i, data in enumerate(data_loader):
                with torch.no_grad():
                    result = self.model(return_loss=False, rescale=True, **data)
                img_metas = data['img_metas'][0].data[0]
                for idx, img_meta in enumerate(img_metas):
                    results_dict[img_meta['filename']] = [None, None, result[idx]]
            # prog_bar.update()
            all_results_dicts[sub_dir] = results_dict

        return all_results_dicts


    def _get_3d_bboxes(self, all_results_dicts, mask_dict):
        # Convert predictions and Merge 2D bounding boxes to form a 3D bounding box

        return boxes_3d, boxes_group


    def run(self, sub_dir_img, mask_dict):
        self.img_root = None
        self.cfg.data.test_ct_all.image_root = None
        # Perform 2D inference
        all_results_dicts = self._infer_2d(sub_dir_img, mask_dict)
        # Extract 3D bounding boxes
        boxes_3d, boxes_group = self._get_3d_bboxes(all_results_dicts, mask_dict)
        return boxes_3d, boxes_group
    

    def inference(self, sub_dir_img, mask_dict):
        det_results, boxes_group = self.run(sub_dir_img, mask_dict)
        
        detect_dict = {}
        box_dict = {}
        
        for sub_dir in det_results:
            detect_dict[sub_dir.replace("_0000.nii.gz", "")] = {
                "x_min":[],
                "y_min":[],
                "z_min":[],
                "x_max":[],
                "y_max":[],
                "z_max":[],
                "score":[]
            }
            seg = mask_dict[sub_dir.replace("_0000.nii.gz", "")]
            for item in det_results[sub_dir]:
                xmin, ymin, zmin, w, h, d, score = item
                xmin, ymin, zmin = [int(x) for x in [xmin, ymin, zmin]]
                xmax = int(xmin+w)
                ymax = int(ymin+h)
                zmax =int(zmin+d-1)
                cls = seg[zmin:zmax+1, ymin:ymax, xmin:xmax].max()
                if cls==0:
                    continue
                detect_dict[sub_dir.replace("_0000.nii.gz", "")]["x_min"].append(xmin)
                detect_dict[sub_dir.replace("_0000.nii.gz", "")]["y_min"].append(ymin)
                detect_dict[sub_dir.replace("_0000.nii.gz", "")]["z_min"].append(zmin)
                detect_dict[sub_dir.replace("_0000.nii.gz", "")]["x_max"].append(xmax)
                detect_dict[sub_dir.replace("_0000.nii.gz", "")]["y_max"].append(ymax)
                detect_dict[sub_dir.replace("_0000.nii.gz", "")]["z_max"].append(zmax)
                detect_dict[sub_dir.replace("_0000.nii.gz", "")]["score"].append(score)
            box_dict[sub_dir.replace("_0000.nii.gz", "")] = boxes_group[sub_dir]
        return detect_dict, box_dict
