import os 
import glob 
import numpy as np 
import SimpleITK as sitk 
from scipy.ndimage import label
from scipy.ndimage.filters import gaussian_filter

import torch 
from liver_seg.model.liverseg import LiverSegNet



class LiverTumorSegmentation:
    def __init__(self, model_folder, folds, checkpoint_name="model_final_checkpoint"):
        self.model_folder = model_folder
        self.folds = folds
        self.checkpoint_name = checkpoint_name
        self.model = LiverSegNet()
        self.pth = torch.load(os.path.join(self.model_folder, self.checkpoint_name), map_location="cpu")["model"]
        self.model.load_state_dict(self.pth)
        self.model.eval()

        # Sliding slice
        self.num_slices
        # Define weights 
        self.weight 


    def predict_cases(self, sub_dir_img):
        """
        sub_dir_img (dict) - Saved data for all phases
        """

        torch.cuda.empty_cache()

        self.model.cuda()
        seg_mask = {}
        
        # Iterate through each key in sub_dir_img
        for key in sub_dir_img:
            image = sitk.GetArrayFromImage(sub_dir_img[key])

            # Normalization image
            image = normalization(image) 

            # Calculate new shape based on spacing

            # Perform interpolation on the image
            image = Interpolate(image) 

            # Check if new shape is less than the required slices


            # Initialize mask, scores, and weights
            mask = np.zeros((old_shape))
            scores = torch.zeros((2, new_shape)).cuda()
            weights = scores*0+1e-6
            
            # Iterate through slices
            for slice_index in range(0, image.shape[2]-self.num_slices//2+1, self.num_slices//2):
                if image.shape[2]-slice_index<self.num_slices+2:
                    slice_index = image.shape[2] - self.num_slices-2
                # Extract image patch
                image_patch = image[:, :, slice_index:slice_index+self.num_slices+2]

                # Forward pass through the model
                with torch.no_grad():
                    pred = self.model(image_patch)[-1]

                    # Resize and Sigmoid

                    # Update scores and weights
                    scores[:, slice_index+1:slice_index+self.num_slices+1] += pred[:, 1:-1]*self.weight
                    weights[:, slice_index+1:slice_index+self.num_slices+1] += self.weight
            
            # Normalize scores
            scores = scores / weights
            # Resize scores
            scores = Interpolate(scores)

            # Create segmentation mask based on thresholding
            mask[scores[0] > 0.5] = 1
            mask[scores[1] > 0.5] = 2

            # Perform post-processing on the mask
            mask = PostProcess(mask)

            # Store the segmentation mask in the dictionary
            seg_mask[key] = mask.astype(np.uint8)
            
        self.model.cpu()
        torch.cuda.empty_cache()
        return seg_mask


    def get_liver_tumor_seg(self, sub_dir_img):
        mask_dict = self.predict_cases(sub_dir_img)
        
        # Write and save to mask_save_path

        torch.cuda.empty_cache()
        return mask_dict

