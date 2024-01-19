# A Multicenter Study of a Clinically Applicable AI System for Automated Detection and Diagnosis of Focal Liver Lesions

## Description
This implementation corresponds to the Liver Artificial Intelligence Diagnosis System (LiAIDS) proposed in the paper *A Multicenter Study of a Clinically Applicable AI System for Automated Detection and Diagnosis of Focal Liver Lesions*. The architecture of the proposed LiAIDS consists of three main modules, namely lesion detection, liver segmentation, and lesion classification modules. The lesion detection module was designed to automatically identify and localize all potential FLL candidates. The liver segmentation module serves as a false positive detector, filtering out lesions detected outside the liver region. Finally, the lesion classification module aims to differentiate the detected lesions into one of the seven most common disease types (i.e., HCC, ICC, HM, FNH, HH, HC, and HA) and further classify them as malignant or benign.


## System Requirements
1. Software Dependencies: The required software dependencies are outlined in the file  ```requirements.txt ```.

2. Compatibility: The code has been rigorously tested on a Linux platform with Python 3.6.8, CUDA 11.2+, PyTorch 1.10.1+, and GPU RTX 2080Ti. 

3. Hardware Requirements: No specialized hardware is needed for this application.

## Code Structure
The initialization  of detection, segmentation, and classification models is implemented in the ```prepare()``` function within ```liver_detector.py```.  All models are used for inference in the ```main_course()``` function. The detection module is found in the Class ```NoduleDet``` in ```get_nodule_det.py```; segmentation  in ```LiverTumorSegmentation``` in ```get_liver_tumor_seg.py```; classification in  ```ClsModel``` in ```get_tumor_cls.py```.
