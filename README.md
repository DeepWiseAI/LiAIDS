# A Multicenter Study of a Clinically Applicable AI System for Automated Detection and Diagnosis of Focal Liver Lesions

## System Requirements
1. Software Dependencies: The required software dependencies are outlined in the file  ```requirements.txt ```.

2. Compatibility: The code has been rigorously tested on a Linux platform with Python 3.6.8, CUDA 11.2+, PyTorch 1.10.1+, and GPU RTX 2080Ti. 

3. Hardware Requirements: No specialized hardware is needed for this application.

## Installation Guide
1. Installation Steps: Execute the commands below:

```
$ pip install -r requirements.txt
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip3 install mmcv-full==1.3.11 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.1/index.html
```
2. Installation Time: The process usually completes within 30 minutes on a standard desktop computer.

## Demo
1. Running the Demo: Execute the following command prediction demos:
```
python LiverDetect/liver_testor.py
```

2. Expected Output: The program will generate a JSON file named  ```liver_pID.json ``` for each pID data, containing diagnostic and detection results.

3. Expected Run Time: Four examples are included, with an anticipated run time of 50 seconds each.

## Instructions for Use

How to Run on Test Samples:
Test samples are structured as follows in the ```data_example```  folder:   
```
data directory structure：
├── pID_1
    ├── studyID
        ├── seriesID_1
            ├── *.dcm 
        ├── seriesID_2
            ├── *.dcm 
        ├── seriesID_3
            ├── *.dcm 
├── pID_2
    ├── studyID
        ├── seriesID_1
            ├── *.dcm 
        ├── seriesID_2
            ├── *.dcm 
        ├── seriesID_3
            ├── *.dcm 
...
```

Clinical information for each patient is also required and provided in  ```LiverDetect/clinical_6label_0425.xlsx```. The table headers include ``` pid ```,  ``` 乙肝```,  ``` 丙肝```,  ``` 肝硬化```,  ``` 肝外肿瘤病史```,  ``` gender```,  and ```age```,  representing the patient ID; histories of  Hepatitis B, Hepatitis C,  and  extrahepatic tumors (0 means negative,1 means positive); gender ( ```女 ``` means female,  ```男 ``` means male); and age of the patient.

Descriptions:

Data consist of at least three phases: plain (non-contrasted) scan phase, arterial phase, and venous phase.

The entry point is ```LiverDetect/liver_testor.py```. 

Input each dataset into the ```subjects``` list as shown below:
```
["root_path/pID1/studyID/plain_seirsID,root_path/pID1/studyID/arterial_seirsID,root_path/pID1/studyID/venous_seirsID","root_path/pID2/studyID/plain_seirsID,root_path/pID2/studyID/arterial_seirsID,root_path/pID2/studyID/venous_seirsID"]
```
All algorithms, including liver tumor segmentation, diagnosis and detection, are called in ```LiverDetect/liver_detector.py```.   

Use the following command for predictions:
```
python LiverDetect/liver_testor.py
```

## Code Structure
The initialization  of detection, segmentation, and classification models is implemented in the ```prepare()``` function within ```liver_detector.py```.  All models are used for inference in the ```main_course()``` function. The detection module is found in the Class ```NoduleDet``` in ```get_nodule_det.py```; segmentation  in``` LiverTumorSegmentation``` in ```get_liver_tumor_seg_2d.py```; classification in  ```ClsModel ``` in ```get_tumor_classification.py```.

## Results Format
Segmentation results are saved in the directory ```liverseg_result```. A JSON file named ```liver_pID.json``` is also saved for each pID data, providing details about each data and the results of diagnosis and detection. The JSON format is as follows:

```
{
    "task": str, "LiverDetect",        
    "quality": str,        
    "json_format_version": str, "2.0.0.180430"         
    "patientId": str, "",                                    //patient id in dicom(0x0010, 0x0020)
    "studyUid": str, "",                                     //study uid in dicom(0x0020, 0x000d)
    "seriesUid": str,                                       //series uid in dicom(0x0020, 0x000e)
    "slice_spacing": float,    
    "slice_thickness": float
    "pixel_spacing": [
            0.683594,
            0.683594
        ],
    "nodes":[
        {
            "GUID": str,"a8098c1a-f86e-11da-bd1a-00112444be1e",          
            "node_index": int, 1,                                //node index
            "score": float, 0.0                                  //detection score
            "score_cls": [                                       //tumor classification score
                    1.9061649254581425e-06,
                    0.0003737528750207275,
                    7.520419603679329e-05,
                    0.9975354671478271,
                    0.0019218528177589178,
                    5.131357102072798e-05,
                    4.045934838359244e-05
                ],
            "label": int, 1                                     //predict label
            "note": str, "",                                             
            "type": str, "Liver_Nodule",                                  
            "attr": {
                    "\u200bthresh": 0.1,
                    "hidden": "false",   
                    "node_type": "FNH",                        // tumor category
                    "density_type": "Sod",
                    "malignant_risk": "LRi",                   // tumor benign and malignant
                    "malignant": 0.002048370584816439          // malignant score
                },
            "series":[     // list for each series
                {
                    "seriesUid":series_uid,
                    "seriesName": "plainScan",                 // plainScan、arterial、portal、late
                    "bounds":[                                 // bounds for each slice
                        {
                            "slice_index": int, 33,            //image instance number in dicom
                            "edge": list/tuple, [[100,100],[200,100],[200,200],[100,200]]  // edge for bounds 
                        },
                        {
                            "slice_index": int, 34,
                            "edge": list/tuple, [[100,100],[200,100],[200,200],[100,200]]
                    }
                    ...
                    ],
                    "rois":[                                         // rois for tumor
                        {
                            "slice_index": int, 33,                  //image instance number in dicom
                            "edge": list/tuple, [[147,338],[147,339],[146,340],[146,340],[147,341],[148,342]]
                        },
                        {
                            "slice_index": int, 34,
                            "edge": list/tuple, [[147,338],[147,339],[146,340],[146,340],[147,341],[148,342]]
                        }
                        ...
                    ],
                    "bbox3d":[center_x, center_y, center_z, w, h, d], // bounds for 3D [center_x, center_y, center_z, w, h, d] 
                }
                {...}                                            // node in other series
            ]
        }
        {...}                                                   // other nodes
    ],
}
```


Descriptions:

```
nodes: A series of tumors for pID data after three-phase registration. 
```
```
nodes/attr/node_type: The tumor prediction type. 
```
```
nodes/scores_cls: The tumor prediction score in the order of ["0": 'Cyst', "1": 'HEM', '2': 'HCC', "3": 'FNH', "4": 'Metastases', "5": 'ICC', "6": 'Abscess'] .
```
```
nodes/series/bounds: 2D detection box in each slice for specific series.
```
```
nodes/series/bbox3d: 3D detection box generate by "bounds."
```
```
nodes/series/rois: Tumor ROIs in each slice for a specific series. 
```
