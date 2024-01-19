
import os.path as op
import numpy as np

# Configuration Class
class Config7Type(object):
    class_dict = {
    "0": '良性',
    "1": '恶性',  
    } 

    sub_class_dict = {
        "0": '肝囊肿',
        "1": '血管瘤',  
        '2': '肝细胞肝癌',
        "3": 'FNH',  
        '4': '肝转移癌',
        '5': '胆管细胞癌',
        '6': '肝脓肿',
    } 

    benign = [0, 1, 3, 6]
    malignant = [2, 4, 5]
    

    #### model #############
    model_name = 'TripleSplitNetHBP_220428'
    weight_path =''
    fusion_weight_path =''
    nchannel = 1          
    n_classes = len(sub_class_dict)

    #### data aug #########################################
    win = []
    final_size_shape = []    
    final_crop_size_shape = []    


    #### data_crop#########################################
    data_crop_name = 'data_crop_v4'
    phase_list = ['Arterial Phase', 'Venous Phase', 'Non Contrast']

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


    def details(self):
        """Display Configuration values."""
        lines=''
        lines += '===============> Configurations:\n'
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                lines += "{:30} {}\n".format(a, getattr(self, a))
        return lines
