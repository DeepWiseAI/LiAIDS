import LiverDetect.utils.rect
import LiverDetect.utils.cube
import LiverDetect.utils.union_find
import numpy as np

class Patch3D(object):
    """
    Patch3D is defined as
    (x, y, z, w, h, d, t(direction), label, score, score1, score2, score3)
    """
    def __init__(self, cube, direct, label, score):
        self.err_str = ''
        self.is_valid = True
        self.cube = cube
        self.direct = direct
        self.label = label
        self.score = score
        if type(direct) != int or direct != 0:
            self.err_str = '[ERROR] direct != 0'
            self.is_valid = False
            return
        if not LiverDetect.utils.cube.check_cube(cube):
            self.err_str = '[ERROR] wrong cube'
            self.is_valid = False
            return
        if label < 0 or score < 0:
            self.err_str = '[ERROR] label < 0 or score < 0'
            self.is_valid = False
            return

    def rescale(self, scales):
        scale_x, scale_y, scale_z = scales
        x, y, z, w, h, d = self.cube
        self.cube = [int(x * scale_x), int(y * scale_y), int(z * scale_z),
                     int(w * scale_x), int(h * scale_y), int(d * scale_z)]


class Bound3D(object):
    """
    Bound3D is used for nodule detection bound.
    ##@member cube: (x, y, z, w, h, d)
    ##@member direct: 0
    ##@member label: label of the detection
    ##@member score: score of the detection
    ##@member is_init: is empty
    ##@member is_patch: is an unit patch. if this is an unit patch, then src_patches=[] and get_patch_num()=1
    ##@member src_patches: the bound3d is combined from a list of bound3d_s,
                            each of which has only 1 unit size in self.direct
    """
    def __init__(self, cube=(-1, -1, -1, -1, -1, -1), direct=0, label=-1, score=-1):
        self.err_str = ''
        self.is_valid = True
        if type(cube) not in [tuple, list] or len(cube) != 6:
            self.err_str = '[ERROR] cube is wrong'
            self.is_valid = False
            return
        for c in cube:
            if type(c) != int and type(c) != float and type(c) != np.float32:
                self.err_str = '[ERROR] cube is wrong'
                self.is_valid = False
                return
        self.cube = cube
        direct = 0
        self.direct = direct
        if type(label) != int:
            self.err_str = '[ERROR] label is not int'
            self.is_valid = False
            return
        self.label = label
        if type(score) != int and type(score) != float:
            self.err_str = '[ERROR] score is not int or float'
            self.is_valid = False
            return
        self.score = score
        self.is_init = (LiverDetect.utils.cube.check_cube(cube) and label >= 0 and score >= 0)
        self.src_patches = []
        if self.is_init:
            self.src_patches.append(Patch3D(cube, direct, label, score))

    def __str__(self):
        s = 'cube: %s  ' % str(self.cube)
        s += 'patch_number=%d, score=%.4f, label=%d' % (self.get_patch_num(), self.score, self.label)
        return s

    def __repr__(self):
        s = 'cube: %s  ' % str(self.cube)
        s += 'patch_number=%d, score=%.4f, label=%d' % (self.get_patch_num(), self.score, self.label)
        return s

    @staticmethod
    def __check_valid(cube, direct, label, score):
        is_direct_valid = (direct == 0)
        return LiverDetect.utils.cube.check_cube(cube) and label >= 0 and score >= 0 and is_direct_valid

    def __combine_score(self, score):
        self.score += score

    def append(self, cube, direct, label, score, label_combine_matrix):
        if self.is_init:
            label_combined = label_combine_matrix[self.label][label]
            if label_combined < 0:
                return
            self.cube = LiverDetect.utils.cube.cube_union(self.cube, cube)
            self.__combine_score(score)
            if label_combined < 0 or label_combined >= label_combine_matrix.shape[0]:
                self.is_valid = False
                return
                # return MError(MError.E_FIELD_BOUND, 2, '[ERROR] wrong label_combined'), None
            self.label = label_combined
        else:
            self.cube = cube
            self.direct = direct
            self.label = label
            self.score = score
            self.is_init = True
        self.src_patches.append(Patch3D(cube, direct, label, score))
        return
        # return MError(MError.E_FIELD_BOUND, 0, ''), True

    def get_score(self):
        return self.score

    # gkl
    def set_avg_score(self, num2d):
        self.avg_score = round(self.score / num2d, 4)

    def get_patch_num(self):
        return len(self.src_patches)

    def __dcm_to_physic_axis(self, point, origin_point, physic_direct, pixel_spacing_3d):
        origin_point = np.array(origin_point)
        # print(point, origin_point, self.pixel_spacing_3d)
        return origin_point + physic_direct * np.array(point) * pixel_spacing_3d

    def __trans_xyz_to_dcm(self, point, pixel_spacing_3d, norm_rate=0.6):
        return np.array(point) * norm_rate / pixel_spacing_3d

    def __convert_to_physics_axis(self, point, pixel_spacing_3d, origin_point,
                               physic_direct, norm_rate=0.6):
        dcm_axis = self.__trans_xyz_to_dcm(point, pixel_spacing_3d, norm_rate)
        # print('dcm axis', dcm_axis)
        physic_axis = self.__dcm_to_physic_axis(dcm_axis, origin_point,
                                                physic_direct, pixel_spacing_3d)
        return physic_axis



    def __is_number(self, a):
        try:
            x = float(a)
            return True
        except:
            return False

def __can_two_combined(b1, b2, label_combine_matrix, least_inter_ratio_matrix):
    """
    :param b1: bound1
    :param b2: bound2
    :param label_combine_matrix: k*k numpy array. If b1/b2 has label l1/l2,
        m[l1][l2] is the label for combined bound.  m[l1, l2]=-1 if cannot be combined.
    :param least_inter_ratio_matrix: we combine 2 bounds only when the
        intersect_area/min_bound_area >= least_inter_ratio_matrix[l1][l2]
    :return: whether two bounds can be combined.
    """
    x1, y1, w1, h1, slice1, l1, s1 = b1
    x2, y2, w2, h2, slice2, l2, s2 = b2
    label_combined = label_combine_matrix[l1][l2]
    least_inter_ratio = least_inter_ratio_matrix[l1][l2]
    if label_combined < 0:
        return False
    inter_rect = LiverDetect.utils.rect.rect_intersect((x1, y1, w1, h1), (x2, y2, w2, h2))
    if not LiverDetect.utils.rect.check_rect(inter_rect):
        return False
    min_area = min(w1 * h1, w2 * h2)
    #iou = (inter_rect[2] * inter_rect[3] * 1.0)/ (w1 * h1 +  w2 * h2 - inter_rect[2] * inter_rect[3])
    if (inter_rect[2] * inter_rect[3] * 1.0 / min_area) < least_inter_ratio:
        return False
    #elif iou < 0.0:
    #    return False
    return True


def combine_bounds_3d_direct(bound2ds_list, opt):
    """
    TODO: think of the score between the
    :param bound2ds_list: list of bounds of all slices. bounds[i] contains all bounds2d in slice_i,
        which is n*6 list array, each row defines as [x, y, w, h, slice_id, label, score]
    :param opt: options for combining
                label_combine_matrix: k*k numpy array. If b1/b2 has label l1/l2,
                        m[l1][l2] is the label for combined bound.  m[l1, l2]=-1 if cannot be combined.
                least_inter_ratio_matrix: we combine 2 bounds only when the
                        intersect_area/min_bound_area >= least_inter_ratio_matrix[l1][l2]
                max_slices_stride: when combined, we can at most skip max_slices_stride-1 slices
                        if the bounds are not continuous
    :return: bound2d
    """
    label_combine_matrix = opt['label_combine_matrix']
    least_inter_ratio_matrix = opt['least_inter_ratio_matrix']
    max_slices_stride = opt['max_slices_stride']

    num_bounds = len(bound2ds_list)
    num_slice = 0
    for bound2d in bound2ds_list:
        num_slice = max(num_slice, bound2d[4])
    num_slice += 1

    # combine all bounds through union-find
    # 1) list all bounds so that each bound can have an id
    bound2ds_ids_by_slice = [[] for i in range(num_slice)]
    for (idx, bound2d) in enumerate(bound2ds_list):
        slice_id = bound2d[4]
        bound2ds_ids_by_slice[slice_id].append({'id': idx, 'bound': bound2d})

    # 2) find all pairs of bounds that can be combined
    combine_pairs = []
    for i in range(num_bounds):
        combine_pairs.append((i, ))
    for slice_id1 in range(num_slice):
        # find pairs inside current slice and current-slice-vs-next-max_slices_stride-slices
        slice_bounds_with_ids1 = bound2ds_ids_by_slice[slice_id1]
        slice_bounds_num1 = len(slice_bounds_with_ids1)
        for slice_id2 in range(slice_id1, min(num_slice, slice_id1 + max_slices_stride + 1)):
            slice_bounds_with_ids2 = bound2ds_ids_by_slice[slice_id2]
            slice_bounds_num2 = len(slice_bounds_with_ids2)
            for i in range(slice_bounds_num1):
                if slice_id1 == slice_id2:
                    j_start = i + 1
                else:
                    j_start = 0
                for j in range(j_start, slice_bounds_num2):
                    b1 = slice_bounds_with_ids1[i]['bound']
                    b2 = slice_bounds_with_ids2[j]['bound']
                    id1 = slice_bounds_with_ids1[i]['id']
                    id2 = slice_bounds_with_ids2[j]['id']
                    if __can_two_combined(b1, b2, label_combine_matrix, least_inter_ratio_matrix):
                        combine_pairs.append((id1, id2))

    # 3) union find
    uf = LiverDetect.utils.union_find.UnionFind(combine_pairs)
    combined_bound_ids = uf.run()

    # 4) combine the bounds
    bound_groups = []
    for bound_ids in combined_bound_ids:
        bound_group = []
        for b_id in bound_ids:
            bound_group.append(bound2ds_list[b_id])
        bound_groups.append(bound_group)

    # May need to check all bounds again
    # TODO

    return bound_groups
