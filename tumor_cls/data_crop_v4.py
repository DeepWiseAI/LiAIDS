import os
import glob

import json
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage

class CropPatch(object):
    def __init__(self, sub_dir_livermask):
        self.need_simplephase_keys = ['A', 'V', 'non']
        self.sub_dir_livermask = sub_dir_livermask

        self.sub_dir_livermask_max = {}

        for sub_dir, livermask_item in self.sub_dir_livermask.items():
            liver_array = livermask_item.swapaxes(0, 2)
            self.sub_dir_livermask_max[sub_dir] = liver_array



    def register(self, df, box_dict):
        """
        Due to the possibility that a tumor may not exhibit obvious features in a certain phase, 
        leading to the inability to detect the corresponding tumor, it is necessary to register the detected phases 
        to those where the tumor was not detected. 

        Args:
        df (pandas) - The basic information of lesions in the data, including spacing, 
        the initial position of the liver, physical location information, etc
        box_dict (dict) - Detection results
        """

    
        #确保行索引唯一，新增的行索引起点，（匹配会新增行）
        # updating_lastindex = len(df)
        updating_lastindex = max(df.index)+1
        df = df.replace(np.nan, '0_', regex=True)
        df = df.replace("None", '0_', regex=True)


        #过滤掉nodule期相sub_dir重复的行
        df = df.drop_duplicates(subset=['sub_dir', 'pid', 'num_id', 'phase'], keep= 'first')
        df = df[df['slice_thickness'] >= 2.]

        df['xmin_scale'] =np.nan
        df['xmax_scale'] =np.nan
        df['ymin_scale'] =np.nan
        df['ymax_scale'] =np.nan
        df['zmin_scale'] =np.nan
        df['zmax_scale'] =np.nan

        df['crop_xmin_register'] =np.nan
        df['crop_xmax_register'] =np.nan
        df['crop_ymin_register'] =np.nan
        df['crop_ymax_register'] =np.nan
        df['crop_zmin_register'] =np.nan
        df['crop_zmax_register'] =np.nan

        #获取肝脏mask相关信息
        df['liver_length'] = df.apply(lambda onerow: abs(onerow['liver_seg_end_slice'] - onerow['liver_seg_begin_slice']), axis=1)

        # Register
        df_register_nodules = df.copy()
        """
        # Loop through nodules id
            # Determine whether the lesion phase is complete, if not continue
            # Retrieve the relative position of the lesion within the liver for phases containing the lesion
               nodule_cropzmin_rs = (crop_zmin - liver_zmin)/(liver_zmax - liver_zmin),
               where crop_zmin is the minimum slice of the lesion along the z-axis, liver_zmin and liver_zmax is 
               the minimum and maximum slice of the liver along the z-axis. You can get nodule_cropzmax_rs, 
               nodule_cropxmin_rs, nodule_cropxmax_rs, nodule_cropymin_rs, nodule_cropymax_rs using the same method

            # Loop lack phase 
                # Based on the relative position of the lesion to the liver in other phases, calculate the position 
                  of the tumor in phases where the lesion is missing.
                  crop_zmin_register = nodule_cropzmin_rs * (liver_zmax_ - liver_zmin_) + liver_zmin,
                  where liver_zmin_ and liver_zmax_ is the minimum and maximum slice of the liver along the z-axis in the lack phase. 
                  You can get other coordinates using the same method
                # Add to the df_register_nodules
            # End Loop
        # End Loop
        """

        return df_register_nodules, box_dict


    def get_coords(self, df_info, sub_dir_img):
        # read excel
        df = df_info.copy(deep=True)
        df['phase'].replace('A', 'Arterial Phase',inplace=True)
        df['phase'].replace('V', 'Venous Phase',inplace=True)
        df['phase'].replace('non', 'Non Contrast',inplace=True)
        df['phase'].replace('D', 'Delay',inplace=True)

        expand_scale = 1.5

        dict_allnodule = {}
        image_array_dict = {}
        for key in sub_dir_img:
            image_array_dict[key] = {}
            image_array_dict[key]['array'] = sitk.GetArrayFromImage(sub_dir_img[key]).swapaxes(0, 2)
            image_array_dict[key]['spacing'] = sub_dir_img[key].GetSpacing()

        for nodule_id in df["num_id"].unique():
            dict_allnodule[nodule_id] = {}
            df_nodule = df[df["num_id"]==nodule_id]

            for row_index in df_nodule.index:
                coords = df_nodule.loc[row_index, ['crop_xmin_register', 'crop_xmax_register', 'crop_ymin_register', 'crop_ymax_register', 'crop_zmin_register', 'crop_zmax_register']].values.astype(np.int32)
                start_x, end_x, start_y, end_y, start_z, end_z = coords
                

                sub_dir_excel = df_nodule.loc[row_index, ["sub_dir"]].values[0]
                real_phase_name = df_nodule.loc[row_index, ['phase']].values[0]
                image_array = image_array_dict[sub_dir_excel]['array']
                spacing_z = image_array_dict[sub_dir_excel]['spacing'][-1]
                expand_z = 1
                if spacing_z < 5:
                    expand_z = 5 / spacing_z
                
                deepth_ = end_z - start_z + 1 
                height_ = (end_y - start_y + 1)
                width_ = (end_x - start_x + 1)
                center_z = (start_z + end_z) / 2
                center_y = (start_y + end_y) / 2 
                center_x = (start_x + end_x) / 2 
                
                deepth = int(deepth_ + int(4 * expand_z))
                height = int(height_ * expand_scale)
                width = int(width_ * expand_scale)

                # If the crop size is less than 64, directly crop with a maximum of (width, height, 64). If it's greater than 64, 
                # extend it outward by 1.5 times. Use centerz as the center, including two slices above and below along the z-axis.
                if width < 64 or height < 64:
                    width = max(width, height, 64)
                    height = width

                min_z = max(min(max((start_z + end_z) / 2 - deepth / 2, 0), image_array.shape[2] - deepth), 0)
                min_y = max(min(max((start_y + end_y) / 2 - height / 2, 0), image_array.shape[1] - height), 0)
                min_x = max(min(max((start_x + end_x) / 2 - width / 2, 0), image_array.shape[0] - width), 0)
                min_z = int(np.round(min_z))
                min_y = int(np.round(min_y))
                min_x = int(np.round(min_x))
                max_z = int(min(np.round(min_z + (center_z - min_z) * 2) + 1, image_array.shape[2]))
                max_y = int(min(np.round(min_y + (center_y - min_y) * 2) + 1, image_array.shape[1]))
                max_x = int(min(np.round(min_x + (center_x - min_x) * 2) + 1, image_array.shape[0]))
                
                image_array = image_array[min_x : max_x, min_y : max_y, min_z : max_z]

                dict_allnodule[nodule_id][real_phase_name] = {}
                dict_allnodule[nodule_id][real_phase_name]['path'] = sub_dir_excel
                dict_allnodule[nodule_id][real_phase_name]['coords_origin'] = [start_x, end_x, start_y, end_y, start_z, end_z]
                dict_allnodule[nodule_id][real_phase_name]['coords_z'] = df_nodule.loc[row_index, ['z_max']].values.astype(np.int32) - df_nodule.loc[row_index, ['z_min']].values.astype(np.int32) 
                dict_allnodule[nodule_id][real_phase_name]['coords'] = [min_x, max_x, min_y, max_y, min_z, max_z]
                dict_allnodule[nodule_id][real_phase_name]['img'] = image_array

        
        return dict_allnodule



class Data_crop_v4:
    def data_crop_func(self, df, sub_dir_img, sub_dir_livermask, box_dict):
        """
        Args:
        df (pandas) - The basic information of lesions in the data, including spacing, 
        the initial position of the liver, physical location information, etc
        sub_dir_img (dict) - Saved data for all phases
        sub_dir_livermask (dict)) - Stored segmentation results of the liver for all data
        box_dict (dict) - Detection results
        """
        croppatch = CropPatch(sub_dir_livermask)
        df_register, box_dict = croppatch.register(df, box_dict)
        nodules = croppatch.get_coords(df_register, sub_dir_img)
        return nodules, box_dict


