import os
import glob
import pydicom
import SimpleITK as sitk

def reverse_dcm(dcm_names):
    instanceNumbers = []
    for dcm_name in dcm_names:
        instanceNumbers.append(pydicom.read_file(dcm_name, stop_before_pixels=True).InstanceNumber)
    return instanceNumbers[0]>instanceNumbers[-1], instanceNumbers

def read_dcm(dcm_folder, save_root="./"):
    reader = sitk.ImageSeriesReader()
    dcm_names = reader.GetGDCMSeriesFileNames(dcm_folder)
    if len(dcm_names)<10:
        raise Exception("dicom series {} less than 10 files: {}".format(dcm_folder, len(dcm_names)))

    reader.SetFileNames(dcm_names)
    try:
        img = reader.Execute()
    except Exception as ex:
        raise Exception("bad dicom series {}: {}")
    reverse_flag, instanceNumber = reverse_dcm(dcm_names)
    instanceNumber.sort()
    if reverse_flag:
        img_array = sitk.GetArrayFromImage(img)
        img_array = img_array[::-1, :, :]
        image = sitk.GetImageFromArray(img_array)
        image.SetSpacing(img.GetSpacing())
        image.SetDirection(img.GetDirection())
        new_origin = list(img.GetOrigin())
        new_origin[2] = new_origin[2] + img.GetSpacing()[2] * (img.GetSize()[2] - 1)
        image.SetOrigin(tuple(new_origin))
        img = image

    dicom_instance = dcm_names[0]
    dcm_data = pydicom.read_file(dcm_names[0], stop_before_pixels=True)
    pid = dcm_data.PatientID
    studyid = dcm_data.StudyInstanceUID
    seriesid = dcm_data.SeriesInstanceUID
    sub_dir = "_".join([pid.strip('\x00'), studyid.strip('\x00'), seriesid.strip('\x00')])
    
    if not os.path.isdir(dcm_folder):
        raise Exception("dicom folder {} not exists".format(dcm_folder))
    
    return sub_dir, instanceNumber, img, dicom_instance
