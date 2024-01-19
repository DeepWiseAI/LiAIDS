import numpy as np


def is_slice_inside(pred_bound_group, gt_bound_group):
    '''return whether center of pred in gt bbox2d.
    Input gt_bound_group: N x gt_bound, with gt_bound as [x,y,w,h,slice_index]
    Input pred_bound_group: N x pred_bound, with gt_bound as [x,y,w,h,slice_index,score]
    '''
    if len(gt_bound_group) <= 0:
        return False
    for pred_slice in pred_bound_group:
        for gt_slice in gt_bound_group:
            # choose bound_group in same slice (with same slice_index).
            if pred_slice[4] == gt_slice[4]:
                center_x = (pred_slice[0]*2 + pred_slice[2])/2.0
                center_y = (pred_slice[1]*2 + pred_slice[3])/2.0
                if is_point_in_bbox2d(gt_slice, [center_x, center_y]):
                    return True
    return False

def is_point_in_bbox2d(bbox2d, point):
    ''' return whether point in bbox2d.
    '''
    return point[0]>= bbox2d[0] and point[0]<= bbox2d[0]+bbox2d[2] \
            and point[1]>= bbox2d[1] and point[1]<= bbox2d[1]+bbox2d[3]

def bbox3d_intersect(b1, b2):
    x1, y1, z1, w1, h1, d1 = b1
    x2, y2, z2, w2, h2, d2 = b2
    if w1 <= 0 or h1 <= 0 or d1 <= 0 or w2 <= 0 or h2 <= 0 or d2 <= 0:
        return -1, -1, -1, 0, 0, 0
    x1_ = x1 + w1
    y1_ = y1 + h1
    z1_ = z1 + d1
    x2_ = x2 + w2
    y2_ = y2 + h2
    z2_ = z2 + d2
    x3 = max(x1, x2)
    y3 = max(y1, y2)
    z3 = max(z1, z2)
    x3_ = min(x1_, x2_)
    y3_ = min(y1_, y2_)
    z3_ = min(z1_, z2_)
    if x3_ <= x3 or y3_ <= y3 or z3_ <= z3:
        return -1, -1, -1, 0, 0, 0
    return x3, y3, z3, x3_ - x3, y3_ - y3, z3_ - z3


def bbox3d_iou(b1, b2):
    x1, y1, z1, w1, h1, d1 = b1
    x2, y2, z2, w2, h2, d2 = b2
    xi, yi, zi, wi, hi, di = bbox3d_intersect(b1, b2)
    inter_volume = wi * hi * di * 1.0
    union_volume = w1 * h1 * d1 + w2 * h2 * d2 - inter_volume
    return inter_volume / union_volume if union_volume >= 1e-5 else 0

def bbox3d_iom(b1, b2):  # divided by smaller one
    x1, y1, z1, w1, h1, d1 = b1
    x2, y2, z2, w2, h2, d2 = b2
    xi, yi, zi, wi, hi, di = bbox3d_intersect(b1, b2)
    inter_volume = wi * hi * di * 1.0
    mins_volume = min(w1 * h1 * d1, w2 * h2 * d2)
    return inter_volume / mins_volume if mins_volume >= 1e-5 else 0

def bbox3d_center_dist2(b1, b2):
    """
    distance square between center of b1 and b2
    """
    x1, y1, z1, w1, h1, d1 = b1
    x2, y2, z2, w2, h2, d2 = b2
    cx1, cy1, cz1 = x1 + w1 * 0.5, y1 + h1 * 0.5, z1 + d1 * 0.5
    cx2, cy2, cz2 = x2 + w2 * 0.5, y2 + h2 * 0.5, z2 + d2 * 0.5
    # GP: actually, center = top + (w-1, h-1, d-1) / 2.0, however, (c1-c2) is correct because the bias is the same(0.5)
    return (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2 + (cz1 - cz2) ** 2

def bbox2d_intersect(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
        return -1, -1, 0, 0
    x1_ = x1 + w1
    y1_ = y1 + h1
    x2_ = x2 + w2
    y2_ = y2 + h2
    x3 = max(x1, x2)
    y3 = max(y1, y2)
    x3_ = min(x1_, x2_)
    y3_ = min(y1_, y2_)
    if x3_ <= x3 or y3_ <= y3:
        return -1, -1, 0, 0
    return x3, y3, x3_ - x3, y3_ - y3
    
def bbox2d_iou(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    inter_x, inter_y, inter_w, inter_h = bbox2d_intersect(b1, b2)
    inter_area = inter_w * inter_h * 1.0
    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union_area if union_area > 1e-5 else 0


def bbox2d_center_dist2(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    cd2 = (x1 + w1 * 0.5 - (x2 + w2 * 0.5)) ** 2 + ((y1 + h1 * 0.5) - (y2 + h2 * 0.5)) ** 2
    return cd2

