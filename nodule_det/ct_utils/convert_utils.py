from .bound_utils import combine_slice_level_pred
import numpy as np
from .eval_classes import TOP_CLASS, name_cls_map, LUNG_PATTERN_DICT

instance_offset_list = []

def convert_pred(results_dicts, thresh_list, minimum_slice_num, max_slices_stride=5, iom_thresh=0.7, post_score_thresh_list=0.2):
    '''Convert pred to all_boxes .
        all_boxes[cls][img] = N 2d-bound-groups, each bound_group contains \
        n bound2ds as (x,y,w,h,slice_idx, label, score).
    '''
    print('converting prediction ...')
    s_count = 0
    all_boxes = [[[] for _ in results_dicts] for _ in range(len(TOP_CLASS)+1)]
    all_boxes_3d = [[[] for _ in results_dicts] for _ in range(len(TOP_CLASS)+1)]
    for idx, results_dict in enumerate(results_dicts):
        bound_groups, bound3ds = combine_slice_level_pred(results_dict, \
                thresh=thresh_list, cls_num=len(TOP_CLASS)+1, \
                max_slices_stride=max_slices_stride, iom_thresh=iom_thresh, iou_thresh3d=post_score_thresh_list)
        for (bound_group, bound3d) in zip(bound_groups, bound3ds): 
            if len(bound_group) < minimum_slice_num:
                s_count += 1
                continue

            cls = bound_group[0][5]
            #if post_score_thresh_list is not None:
            #    if get_score_one(bound_group) >= post_score_thresh_list[cls]:
            #        all_boxes[cls][idx].append(bound_group)
            #else:
            all_boxes[cls][idx].append(bound_group)
            
            cls = bound3d.label
            score = bound3d.avg_score
            cube = list(bound3d.cube)
            cube.extend([score])
            all_boxes_3d[cls][idx].append(cube)

    all_boxes = np.array(all_boxes)
    #for boxes_im in all_boxes:
        #print(sum([len(boxes) for boxes in boxes_im]))
    print("Total ignored bbox under the size of %d is %d" % (minimum_slice_num, s_count))
    return all_boxes, all_boxes_3d

