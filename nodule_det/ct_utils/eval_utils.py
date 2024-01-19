import numpy as np
import cv2
import os
import os.path as osp
import shutil
import math
import pandas as pd
from PIL import Image,ImageFont,ImageDraw
import textwrap
import SimpleITK as sitk
import pdb

from nodule_det.ct_utils.eval_classes import TOP_CLASS, name_cls_map
from nodule_det.ct_utils.bbox import  is_slice_inside, bbox3d_iou

# sorted to keep order.
name_cls_list = sorted(name_cls_map.items(), key=lambda v:v[1])
# pil_font = ImageFont.truetype("eval_tools/ct_utils/simhei.ttf", 16)

def get_image_dict(bound_groups):
    """Get a image_dict {image_id: [(bbox, label), ...]} from bound_groups"""
    image_dict = {idx : [] for idx in range(len(bound_groups[0]))}
    for cls_idx, bound_group in enumerate(bound_groups):
        if bound_group == []:
            continue
        for img_idx, bounds in enumerate(bound_group):
            if bounds==[]:
                continue
            for bound in bounds:
                image_dict[img_idx].append((bound, cls_idx+1))
    return image_dict

def assign_label_for_pred(pred_bound, gt_bounds):
    """Assign pred_bound with correct labels. 
       Label is 0, i.e. background is pred_bound can not match any gt_bounds.
       pred_bound is a bbox and gt_bounds are list of (bbox, label)
    """
    assigned_label = 0
    max_iou = 0
    for gt_bound in gt_bounds:
        if gt_bound[1] != pred_bound[1]:
            continue
        is_match, iou3d = get_match_status(pred_bound, gt_bound[0])
        if is_match and iou3d > max_iou:
            assigned_label = gt_bound[1]
            max_iou = iou3d
    return assigned_label


def evaluate_pred_image(pred_bounds, gt_bounds):
    """Get the corresponding groundtruth lable for each pred in each image.
       pred_bounds & gt_bounds are list of (bound, cls) 
       return evaluated_bounds as list of (bound, pred_cls, gt_cls)
    """
    new_pred_bounds = []
    for idx, pred_bound in enumerate(pred_bounds):
        assigned_label = assign_label_for_pred(pred_bound[0], gt_bounds)
        new_pred_bounds.append((pred_bound[0], pred_bound[1], assigned_label))
        
    return new_pred_bounds

def filter_thresh(vis_dict, score_th):
    mis_pred = vis_dict['mis_pred']
    mis_pred_filtered = [[] for i in range(len(mis_pred))]
    for idx, bound_groups in enumerate(mis_pred):
        try:
            scores = np.array(get_score(bound_groups))
            for score, bound_group in zip(scores, bound_groups):
                if score > score_th:
                    mis_pred_filtered[idx].append(bound_group)
        except Exception as e:
            print('get error ', e)
            pdb.set_trace()
    vis_dict['mis_pred'] = mis_pred_filtered

def get_match_status(pred_bound_group, gt_bound_group):
    """
    The detect and nodule are matched when:
        'iou3d': 3d IOU >= threshould
        'sliceInside':
    """
    if is_slice_inside(pred_bound_group, gt_bound_group):
        detect_bbox3d = bound_group_to_cube(pred_bound_group)
        nodule_bbox3d = bound_group_to_cube(gt_bound_group)
        iou3d = bbox3d_iou(detect_bbox3d, nodule_bbox3d)
        return True, iou3d
    else:
        return False, 0


def is_match(pred_bound_group, gt_bound_group, metric='sliceInside', iou_thresh=0.001):
    """
    The detect and nodule are matched when:
        'iou3d': 3d IOU >= threshould
        'sliceInside':
    """
    assert metric in ['iou3d','sliceInside'], 'Unknown metric: ' + metric
    if metric == 'iou3d':
        assert 1 >= iou_thresh >= 0
        detect_bbox3d = bound_group_to_cube(pred_bound_group)
        nodule_bbox3d = bound_group_to_cube(gt_bound_group)
        iou3d = bbox3d_iou(detect_bbox3d, nodule_bbox3d)
        return iou3d >= iou_thresh
    if metric == 'sliceInside':
        return is_slice_inside(pred_bound_group, gt_bound_group)
    #if metric == 'center3d':
        #center_dist2 = nodule.get_center_dist_mm2(detect, verbose=get_debug())
        #nodule_radius = nodule.get_radius()
        #return center_dist2 < nodule_radius ** 2

def get_center_mean(score_list, idx):
    """Calculate mean of previous, current and next score."""
    score_sum = 0
    count = 0
    for i in range(3):
        if idx + (i - 1) not in range(len(score_list)):
            continue
        else:
            score_sum += score_list[idx + (i - 1)]
            count += 1
    avg = score_sum / (count * 1.0)
    return avg

def get_moving_avg(avg_dict):
    """Get the 'moving avg' for {'slice_index': score, ...} """
    slice_list = list(avg_dict.keys())
    slice_list.sort()
    sorted_score_list = [avg_dict[idx] for idx in slice_list]
    moving_sum = 0
    moving_max = 0
    for idx in range(len(sorted_score_list)):
        current_score = get_center_mean(sorted_score_list, idx)
        moving_sum += current_score
        if moving_max <= current_score:
            moving_max = current_score
    moving_avg = moving_sum / len(sorted_score_list)
    return moving_avg, moving_max

def get_score(bound_groups_all, moving_avg=False):
    '''Get score for all bound groups
    '''
    score = []
    for bound_group in bound_groups_all:
        score_count = 0.0
        avg_dict = {}
        for bound in bound_group:
            avg_dict[bound[4]] = bound[-1]
            score_count += bound[-1]
        if moving_avg:
            moving_avg_score, moving_max_score = get_moving_avg(avg_dict)
            score.append(moving_avg_score)
        else:
            score.append(score_count/len(bound_group))
    return score

def get_score_one(bound_groups, moving_avg=False):
    '''Get score for all bound groups
    '''
    if True:
        score_count = 0
        avg_dict = {}
        for bound in bound_groups:
            avg_dict[bound[4]] = bound[-1]
            score_count += bound[-1]
        if moving_avg:
            moving_avg_score, moving_max_score = get_moving_avg(avg_dict)
            score = moving_avg_score
        else:
            score = score_count/len(bound_groups)
    return score

def bounds_to_group(bounds, cls, instance_begin):
    '''Convert one gt nodes to bound2d group.
    '''
    group = []
    for bound in bounds:
        slice_index = bound['slice_index']-instance_begin
        edge = bound['edge']
        x1 = bound['edge'][0][0]
        y1 = bound['edge'][0][1]
        x2 = bound['edge'][2][0]
        y2 = bound['edge'][2][1]
        group.append([x1, y1, x2-x1+1, y2-y1+1, slice_index, cls])
    return group

def get_bounds_size(bounds):
    max_size = 0
    for bound in bounds:
        edge = bound['edge']
        x1 = bound['edge'][0][0]
        y1 = bound['edge'][0][1]
        x2 = bound['edge'][2][0]
        y2 = bound['edge'][2][1]
        cur_size = max(y2-y1, x2-x1)
        if cur_size > max_size:
            max_size = cur_size
    return max_size

def bounds_to_cube(bounds):
    '''Find maximum adjacent cube for GT bounds.
    '''
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    for bound in bounds:
        x1.append(bound['edge'][0][0])
        y1.append(bound['edge'][0][1])
        x2.append(bound['edge'][2][0])
        y2.append(bound['edge'][2][1])
    z1 = bounds[0]['slice_index']
    z2 = bounds[-1]['slice_index']
    cube = [min(x1), min(y2), z1,
            max(x2)-min(x1)+1, max(y2)-min(y1)+1, z2-z1+1]
    return cube

def bound_group_to_cube(bound_group):
    '''Find maximum adjacent cube for GT or Pred bound_group.
    '''
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    slice_index = []
    for bound in bound_group:
        x1.append(bound[0])
        y1.append(bound[1])
        x2.append(bound[0]+bound[2]-1)
        y2.append(bound[1]+bound[3]-1)
        slice_index.append(bound[4])
    cube = [min(x1), min(y2), min(slice_index),
            max(x2)-min(x1)+1, max(y2)-min(y1)+1, \
            max(slice_index)-min(slice_index)+1]
    return cube

def filter_det(ct_det_dicts, gt_anns):
    '''Filter bound_groups and gt_anns to preserve overlapped subset.
    Args:
            |
            L__ results_dicts: dict, detection results of CT samples.
            L__ all_image_paths: str, path to image_tensor,'patientID/seriesUID/studyUID'
        gt_anns: dict, Annotations of CT samples. Note that several gt_ann is not in roidbs.

       1. get preserve index for roidbs which exists in gt_anns.
       2. filter bound_groups according to index.
       3. convert gt_annds to gt_bound_groups.
    '''
    results_dicts_filtered = []
    gt_filtered = []
    image_path_filtered = []
    #all_images = ct_det_dicts['all_image_paths']
    #results_dicts = ct_det_dicts['all_results_dicts']
    for gt_ann in gt_anns:
        #for image_path, results_dict in zip(all_images, results_dicts):
        for image_path, results_dict in ct_det_dicts.items():
            patientID, studyUID, seriesUID = image_path.split(os.sep)[-4:-1]
            if patientID==gt_ann['patientID'] and studyUID==gt_ann['studyUID'] \
                    and seriesUID==gt_ann['seriesUID']:
                results_dicts_filtered.append(results_dict)
                gt_filtered.append(gt_ann)
                image_path_filtered.append(image_path)
    print('load {} CT detections and {} labeled CTs, overlapped items among them are {}'\
         .format(len(ct_det_dicts), len(gt_anns), len(gt_filtered)))
    return results_dicts_filtered, gt_filtered, image_path_filtered


def nii_filter_det(ct_det_dicts, gt_anns):
    '''Filter bound_groups and gt_anns to preserve overlapped subset.
    Args:
            |
            L__ results_dicts: dict, detection results of CT samples.
            L__ all_image_paths: str, path to image_tensor,'patientID/seriesUID/studyUID'
        gt_anns: dict, Annotations of CT samples. Note that several gt_ann is not in roidbs.

       1. get preserve index for roidbs which exists in gt_anns.
       2. filter bound_groups according to index.
       3. convert gt_annds to gt_bound_groups.
    '''
    results_dicts_filtered = []
    gt_filtered = []
    image_path_filtered = []
    for gt_ann in gt_anns:
        for image_path, results_dict in ct_det_dicts.items():
            sub_dir = image_path.split(os.sep)[-2]
            if gt_ann['sub_dir'] == sub_dir:
                results_dicts_filtered.append(results_dict)
                gt_filtered.append(gt_ann)
                image_path_filtered.append(image_path)
    print('load {} CT detections and {} labeled CTs, overlapped items among them are {}'\
         .format(len(ct_det_dicts), len(gt_anns), len(gt_filtered)))
    return results_dicts_filtered, gt_filtered, image_path_filtered

def filter_json_gt(json_list, gt_anns):
    '''Filter bound_groups from json and gt_anns to preserve overlapped subset.
    Args:
            |
            L__ json_list: list of json_dict. AI results. 
            L__ all_image_paths: str, path to image_tensor,'patientID/seriesUID/studyUID'
        gt_anns: dict, Annotations of CT samples. Note that several gt_ann is not in roidbs.

    '''
    json_list_filtered = []
    gt_filtered = []
    image_path_filtered = []
    #all_images = ct_det_dicts['all_image_paths']
    #results_dicts = ct_det_dicts['all_results_dicts']
    for gt_ann in gt_anns:
        for json_dict in json_list:
            patientID, studyUID, seriesUID = json_dict['patientID'].strip() , json_dict['studyUID'], json_dict['seriesUID']
            if patientID==gt_ann['patientID'] and studyUID==gt_ann['studyUID'] \
                    and seriesUID==gt_ann['seriesUID']:
                json_list_filtered.append(json_dict)
                gt_filtered.append(gt_ann)
                image_path_filtered.append(osp.join(patientID, studyUID, seriesUID))
    print('load {} CT detections and {} labeled CTs, overlapped items among them are {}'\
         .format(len(json_list_filtered), len(gt_anns), len(gt_filtered)))
    return json_list_filtered, gt_filtered, image_path_filtered

def filter_det_old(ct_det_dicts, gt_anns):
    '''Filter bound_groups and gt_anns to preserve overlapped subset.
    Args:
            |
            L__ results_dicts: dict, detection results of CT samples.
            L__ all_image_paths: str, path to image_tensor,'patientID/seriesUID/studyUID'
        gt_anns: dict, Annotations of CT samples. Note that several gt_ann is not in roidbs.

       1. get preserve index for roidbs which exists in gt_anns.
       2. filter bound_groups according to index.
       3. convert gt_annds to gt_bound_groups.
    '''
    results_dicts_filtered = []
    gt_filtered = []
    image_path_filtered = []
    all_images = ct_det_dicts['all_image_paths']
    results_dicts = ct_det_dicts['all_results_dicts']
    for gt_ann in gt_anns:
        for image_path, results_dict in zip(all_images, results_dicts):
            patientID, studyUID, seriesUID = image_path.split(os.sep)[-4:-1]
            if patientID==gt_ann['patientID'] and studyUID==gt_ann['studyUID'] \
                    and seriesUID==gt_ann['seriesUID']:
                results_dicts_filtered.append(results_dict)
                gt_filtered.append(gt_ann)
                image_path_filtered.append(image_path)
    print('load {} CT detections and {} labeled CTs, overlapped items among them are {}'\
         .format(len(results_dicts), len(gt_anns), len(gt_filtered)))
    return results_dicts_filtered, gt_filtered, image_path_filtered

def filter_det_backup(results_dicts, roidbs, gt_anns):
    '''Filter bound_groups and gt_anns to preserve overlapped subset.
    Args:
        results_dicts: detection results of roidbs, so they have same order.
        gt_anns: a subset of roidbs, while several gt_ann is not in roidbs.
       1. get preserve index for roidbs which exists in gt_anns.
       2. filter bound_groups according to index.
       3. convert gt_annds to gt_bound_groups.
    '''
    results_dicts_filtered = []
    gt_filtered = []
    roidb_filtered = []
    for gt_ann in gt_anns:
        for idx,roidb in enumerate(roidbs):
            import pdb
            patientID, studyUID, seriesUID = roidb['image'].split(os.sep)[-4:-1]
            if patientID==gt_ann['patientID'] and studyUID==gt_ann['studyUID'] \
                    and seriesUID==gt_ann['seriesUID']:
                results_dicts_filtered.append(results_dicts[idx])
                gt_filtered.append(gt_ann)
                roidb_filtered.append(roidb)
    print('load {} CT detections and {} labeled CTs, overlapped items among them are {}'\
         .format(len(roidbs), len(gt_anns), len(gt_filtered)))
    return results_dicts_filtered, gt_filtered, roidb_filtered

# def PIL_text(image, box, text, color):
#     ''' Put text on cv2 image(ndarray) with PIL.
#     '''
#     pil_im = Image.fromarray(np.uint8(image))
#     draw = ImageDraw.Draw(pil_im)
#     x = int(box[0])
#     y = int(box[1]-14)
#     twidth, theight = draw.textsize(text, font=pil_font)
#     draw.rectangle(((x,y),(x+twidth,y+theight)), fill=color[::-1]) # text background color
#     draw.text((x,y), text, (255,255,255), font=pil_font)
#     image = np.array(pil_im)
#     return image

# def draw_on_img(img_path, image_tensor, gt_slice_info, pred_slice_info, draw_info, \
#         instance_offset, out_dir):
#     ''' save visualization results for slice in one CT(image_tensor).
#     '''
#     def draw_box(image, boxes):
#         # gt box   = [x,y,w,h, group_id, label, is_match]
#         # pred box = [x,y,w,h, group_id, label, is_match, score]
#         for box in boxes:
#             if len(box) == 7: # gt box.
#                 text = str(box[4])+str(name_cls_list[box[5]-1][0])
#                 color = (0,139,0) if box[6] else (255,0,0) # green for hit, blue for mis
#             elif len(box) == 8: # pred box, add score.
#                 text = str(box[4])+str(name_cls_list[box[5]-1][0])+str(box[7])[:4]
#                 color = (0,0,255) if box[6] else (255,0,0) # red for TP, blue for FP
#             cv2.rectangle(image, (int(box[0]), int(box[1])), \
#                     (int(box[0]+box[2]), int(box[1]+box[3])), color, 1)
#             image = PIL_text(image, box, text, color[::-1])
#         return image

    # def draw_info_image(im_size):
    #     image = Image.new('RGB', im_size, (0, 0, 0))
    #     draw = ImageDraw.Draw(image)
    #     hit_gt_list, mis_gt_list, hit_pred_list, mis_pred_list = draw_info
    #     gt_text = 'GT(%d hit/%d mis)\n'%(len(hit_gt_list), len(mis_gt_list))
    #     pred_text = 'Pred(%d tp/%d fp)\n'%(len(hit_pred_list), len(mis_pred_list))
    #     for mis_gt in mis_gt_list:
    #         gt_text += '\n[%d,%d,%d]'%(mis_gt[0]-instance_offset, mis_gt[1], mis_gt[2])
    #     for mis_pred in mis_pred_list:
    #         pred_text += '\n[%d,%d,%d]'%(mis_pred[0], mis_pred[1], mis_pred[2])

    #     draw.text((5,5), gt_text, (255,255,255), font=pil_font)
    #     text_x = 0
    #     for idx, line in enumerate(pred_text.split('\n')):
    #         if idx%25 == 0:
    #             text_x += 135
    #         text_y = idx % 25 * 16 +5
    #         draw.text((text_x, text_y), line, (255,255,255), font=pil_font)
    #     return image
    # def add_pred_text(image, bounds):
    #     text = ''
    #     for bound in bounds:
    #         text += str(bound[4])+str(name_cls_list[bound[5]-1][0])+' '
    #     pil_im = Image.fromarray(np.uint8(image))
    #     draw = ImageDraw.Draw(pil_im)
    #     draw.text((5,40), text, (255,255,255), font=pil_font)
    #     image = np.array(pil_im)
    #     return image


    # patientID, studyUID, seriesUID, img_name = img_path.split(os.sep)[-4:]
    # # TODO hard code for liver
    # save_dir = os.path.join(out_dir, studyUID, seriesUID, img_name)
    # #save_dir = os.path.join(out_dir, patientID, studyUID, seriesUID)
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # info_image = draw_info_image(image_tensor.shape[1:])
    # for z_idx in range(image_tensor.shape[0]):
    #     gt_bounds = gt_slice_info[z_idx]
    #     pred_bounds = pred_slice_info[z_idx]
    #     if gt_bounds == [] and pred_bounds == []:
    #         continue
    #     image = image_tensor[z_idx]
    #     image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    #     gt_image = draw_box(image.copy(), gt_bounds)
    #     pred_image = draw_box(image.copy(), pred_bounds)
    #     pred_image = add_pred_text(pred_image, pred_bounds)
    #     cv2.putText(image, 'origin', (5,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #     cv2.putText(gt_image, 'GT slice %d'%z_idx, (5,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)
    #     cv2.putText(pred_image, 'pred', (5,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #     result_img = np.vstack((np.hstack((gt_image,pred_image)),np.hstack((image, info_image))))
    #     save_path = os.path.join(save_dir, str(z_idx)+'.png')
    #     cv2.imwrite(save_path, result_img)

def write_multiple_txt(gt_multi_hit_flag, image_path, save_dir):
    with open(os.path.join(save_dir,'multiple_hit.txt'), 'w') as f:
        for idx, flag in enumerate(gt_multi_hit_flag):
            if flag > 0:
                f.write(image_path[idx] + '\t' + str(flag))

def adjust_ww_wl(image, ww = 510, wc = 45, is_uint8 = True):
    """
    adjust window width and window center to get proper input
    """
    min_hu = wc - (ww/2)
    max_hu = wc + (ww/2)
    new_image = np.clip(image, min_hu, max_hu)#np.copy(image)
    if is_uint8:
        new_image -= min_hu
        new_image = np.array(new_image / ww * 255., dtype = np.uint8)
    return new_image

def save_and_draw(image_paths, vis_dict_list, instance_offset_list, is_draw, out_dir):
    ''' write multiple hit results, and visualize all CTs.
    '''
    # gt_multi_hit_count (pred_multi_hit_count) are np array,
    # with length == len(images), and each elements in it indicate
    # times of multiple hits for corresponding image.
    hit_gt_all, mis_gt_all, hit_pred_all, mis_pred_all, \
        gt_multi_hit_count, pred_multi_hit_count = merge_vis_dict(vis_dict_list)
    # write multiple hit results
    with open(os.path.join(out_dir, 'multiple_hit.txt'), 'w') as f:
        f.write('image_path\t gt_multiple_hit\t pred_multiple_hit\t num_mis_gt\t num_mis_pred\n')
        for img_path, mis_gt, mis_pred, gt_multi_hit, pred_multi_hit in zip( \
            image_paths, mis_gt_all, mis_pred_all, gt_multi_hit_count, pred_multi_hit_count):
            num_mis_gt = sum([len(obj) for obj in mis_gt])
            num_mis_pred = sum([len(obj) for obj in mis_pred])
            sub_dir = os.path.join(*(img_path.split('/')[-4:-1]))
            save_str = '%s\t%s\t%s\t%s\t%s\n' % (sub_dir, gt_multi_hit, \
                    pred_multi_hit, num_mis_gt, num_mis_pred)
            f.write(save_str)

    # draw visulization results.
    if not is_draw:
        return
    vis_save_dir = os.path.join(out_dir, 'vis_ct')
    if os.path.exists(vis_save_dir):
        shutil.rmtree(vis_save_dir, True)
    for img_path, hit_gt, mis_gt, hit_pred, mis_pred, instance_offset in zip( \
         image_paths, hit_gt_all, mis_gt_all, hit_pred_all, mis_pred_all, instance_offset_list):
        # TODO: hard code for liver
        img_path = img_path.replace('/norm_image.npz','')
        image_tensor = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        image_tensor = adjust_ww_wl(image_tensor)
        #image_tensor = np.load(open(img_path, 'rb'))['data']
        z_max = image_tensor.shape[0]
        gt_slice_info, pred_slice_info, draw_info = \
                convert_slice_info(hit_gt, mis_gt, hit_pred, mis_pred, z_max)
        # draw_on_img(img_path, image_tensor, gt_slice_info, pred_slice_info, \
        #         draw_info, instance_offset, vis_save_dir)
    return

def convert_slice_info(hit_gt, mis_gt, hit_pred, mis_pred, z_max):
    ''' Distribute all bounds to corresponding z-slice.
    Input z_max: num of z-axis slices.
    '''
    def append_bounds(slice_info, results, group_id, is_match):
        group_list = []
        for cls,bound_groups in enumerate(results):
        # class->bound_group->bound_2d
            for bound_group in bound_groups:
                group_id += 1
                #for bound_2d in bound_group:
                    #group_list.append((group_id, bound_2d[4])) # (group_id, begin_slice)
                    #break
                try:
                    # (group_id, begin_slice, end_slice)
                    group_list.append((group_id, bound_group[0][4],bound_group[-1][4]))
                except Exception as e:
                    print('error ', e)
                    pdb.set_trace()
                for bound_2d in bound_group:
                    # slice index of bound_2d in one CT
                    slice_idx = bound_2d[4]
                    label = bound_2d[5]
                    if len(bound_2d)==7:
                    # pred bound_2d
                        score = bound_2d[6]
                        slice_info[slice_idx].append(bound_2d[:4] + [group_id,label,is_match,score])
                    elif len(bound_2d)==6:
                    # gt bound_2d
                        slice_info[slice_idx].append(bound_2d[:4] + [group_id,label,is_match])
                    else:
                        pdb.set_trace()
        return group_id, group_list
    group_id = -1
    gt_slice_info = [[] for z_idx in range(z_max)]
    pred_slice_info = [[] for z_idx in range(z_max)]
    group_id,hit_gt_list = append_bounds(gt_slice_info, hit_gt, group_id, True)
    group_id,mis_gt_list = append_bounds(gt_slice_info, mis_gt, group_id, False)
    group_id = -1
    group_id,hit_pred_list = append_bounds(pred_slice_info, hit_pred, group_id, True)
    group_id,mis_pred_list = append_bounds(pred_slice_info, mis_pred, group_id, False)
    draw_info = [hit_gt_list, mis_gt_list, hit_pred_list, mis_pred_list]
    return gt_slice_info, pred_slice_info, draw_info

def merge_vis_dict(vis_dict_list):
    ''' Merge dicts and convert cls-img-groups order to img-cls-groups.
    '''
    hit_gt_all = []
    mis_gt_all = []
    hit_pred_all = []
    mis_pred_all = []
    gt_multi_hit_count_all = []
    pred_multi_hit_count_all = []
    for vis_dict in vis_dict_list:
        hit_gt_all.append(list(vis_dict['hit_gt']))
        mis_gt_all.append(list(vis_dict['mis_gt']))
        hit_pred_all.append(list(vis_dict['hit_pred']))
        mis_pred_all.append(list(vis_dict['mis_pred']))
        gt_multi_hit_count_all.append(vis_dict['gt_multi_hit_count'])
        pred_multi_hit_count_all.append(vis_dict['pred_multi_hit_count'])
    gt_multi_hit_count = sum(gt_multi_hit_count_all)
    pred_multi_hit_count = sum(pred_multi_hit_count_all)
    # 2-dim list transpose 
    hit_gt_all = list(map(list, zip(*hit_gt_all)))
    mis_gt_all = list(map(list, zip(*mis_gt_all)))
    hit_pred_all = list(map(list, zip(*hit_pred_all)))
    mis_pred_all = list(map(list, zip(*mis_pred_all)))

    return hit_gt_all, mis_gt_all, hit_pred_all, mis_pred_all, \
            gt_multi_hit_count, pred_multi_hit_count

def write_excel(df, out_dir, sheetname='Sheet1', write_index=False, file_name='CT_eval.xlsx'):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fname = os.path.join(out_dir, file_name)
    writer = pd.ExcelWriter(fname, engine='xlsxwriter')
    df.to_excel(writer, sheetname, index=write_index)
    writer.save()


