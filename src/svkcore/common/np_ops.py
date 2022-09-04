# -*- encoding: utf-8 -*-

# @File    : np_ops.py
# @Time    : 2019-12-04
# @Author  : zjh

r"""
Some common numpy operations
"""

import numpy as np
import cv2


def ndarray_index(shape):
    """ Create np.ndarray index """
    idx = [np.arange(x) for x in shape]
    if len(idx) >= 2:
        idx[0], idx[1] = idx[1], idx[0]
    idx = np.meshgrid(*idx)
    if len(idx) >= 2:
        idx[0], idx[1] = idx[1], idx[0]
    return np.stack(idx, -1)


def points_distance(points0, points1, weights=(1., 1.)):
    """ Calculate  distances between two point array """
    points0 = np.reshape(points0, [-1, 1, 2])
    points1 = np.reshape(points1, [1, -1, 2])
    dist = ((points0 - points1) * np.asarray(weights)) ** 2
    dist = np.sum(dist, axis=-1)
    dist = np.sqrt(dist)
    return dist


def ellipse_kernel(ksize, dtype=np.int):
    """ Create ellipse kernel """
    assert len(ksize) == 2 and all(x > 0 for x in ksize)

    indexs = ndarray_index(ksize) + 0.5
    center = np.asarray(ksize) / 2
    dis = points_distance(indexs, center, weights=[1, ksize[0] / ksize[1]])
    dis = np.reshape(dis, indexs.shape[:-1])
    kernel = np.zeros(ksize, dtype=dtype)
    kernel[dis <= (ksize[0] - 1) / 2] = 1
    return kernel


def circle_kernel(diameter, dtype=np.int):
    """ Create circel kernel """
    return ellipse_kernel([diameter, diameter], dtype)


def nms_mask(seg, ksize=3, dtype=None):
    h, w = seg.shape
    offset = ksize // 2
    T = np.zeros([x + ksize for x in seg.shape], dtype=seg.dtype)
    T[offset:offset + h, offset:offset + w] = seg
    K = (T[:, i:i + w] for i in range(ksize))
    K = np.stack(K, axis=-1)
    max_x = np.max(K, axis=-1)
    Q = (max_x[i:i + h, :] for i in range(ksize))
    Q = np.stack(Q, axis=-1)
    max_xy = np.max(Q, axis=-1)
    # eq = np.less_equal(np.abs(seg - max_xy), 1e-8)
    eq = np.equal(seg, max_xy)
    if dtype:
        mask = eq.astype(dtype)
    else:
        mask = eq.astype(seg.dtype)
    return mask


def generate_grid(panel_size,
                  grid_size=None,
                  grid_num=None,
                  overlap_size=None,
                  overlap_ratio=None,
                  allow_cross_boundary=False):
    """ Generate grid for crop image patches

    :param panel_size: a tuple of image size (height, width)
    :param grid_size: a tuple of grid size (grid_height, grid_width)
    :param grid_num: a tuple of grid num (grid_rows, grid_column)
    :param overlap_size: a tuple of overlap size
    :param overlap_ratio: a tuple of grid overlap ratio
    :param allow_cross_boundary: allow the last row or column position cross panel or not
    :return: grids [rows, columns, 4], each grid consists (ymin, xmin, ymax, xmax)
    """
    if grid_size is None and grid_num is None:
        raise ValueError("param grid_size/grid_num must be set one")
    elif grid_size is not None and grid_num is not None:
        raise ValueError("param grid_size/grid_num must be set only one")

    if overlap_size is None and overlap_ratio is None:
        raise ValueError("param overlap_size/overlap_ratio must be set one")
    elif overlap_size is not None and overlap_ratio is not None:
        raise ValueError("param overlap_size/overlap_ratio must be set only one")
    elif overlap_ratio is not None:
        if not all(0 <= x < 1 for x in overlap_ratio):
            raise ValueError("param overlap_ratio must between in [0, 1)")
    elif overlap_size is not None and grid_size is not None:
        if not all(0 <= x < y for x, y in zip(overlap_size, grid_size)):
            raise ValueError("param overlap_size must be smaller thangrid_size ")

    if grid_num and overlap_ratio:
        rows, columns = grid_num
        r = rows - (rows - 1) * overlap_ratio[0]
        c = columns - (columns - 1) * overlap_ratio[1]
        grid_size = panel_size[0] / r, panel_size[1] / c
        overlap_size = grid_size[0] * overlap_ratio[0], grid_size[1] * overlap_ratio[1]
    elif grid_num and overlap_size:
        rows, columns = grid_num
        grid_size = (panel_size[0] + (rows - 1) * overlap_size[0]) / rows, \
                    (panel_size[1] + (columns - 1) * overlap_size[1]) / columns
    elif grid_size and overlap_ratio:
        overlap_size = grid_size[0] * overlap_ratio[0], grid_size[1] * overlap_ratio[1]

    # check grid param valid
    grid_size = [int(round(x)) for x in grid_size]
    assert all(x >= 1 for x in grid_size)
    overlap_size = [int(round(x)) for x in overlap_size]
    assert all(x >= 0 for x in overlap_size)
    assert all(0 <= x < y for x, y in zip(overlap_size, grid_size))

    ymin = np.arange(0, panel_size[0], grid_size[0] - overlap_size[0])
    xmin = np.arange(0, panel_size[1], grid_size[1] - overlap_size[1])
    xmin, ymin = np.meshgrid(xmin, ymin)
    ymax, xmax = ymin + grid_size[0], xmin + grid_size[1]

    grids = np.stack([ymin, xmin, ymax, xmax], axis=-1)

    if allow_cross_boundary is False:
        crs_idx = np.where(ymax[:, 0] > panel_size[0])[0]
        if len(crs_idx) > 0:
            grids = grids[:crs_idx[0], :]
        crs_idx = np.where(xmax[0, :] > panel_size[1])[0]
        if len(crs_idx) > 0:
            grids = grids[:, :crs_idx[0]]

    return grids


def seg2point(seg, max_diameter: int, min_distance, fb_threshold: float = 0.5,
              min_fore_count: int = 1, max_fore_count: int = -1,
              avg_fore_score: float = 0.55, distance_weights=(1., 1.)):
    """ Find all valid points from segmentation

    :param seg: point object segmentation result, probability map between [0.0, 1.0]
    :param max_diameter: max_radius for a point
    :param min_distance: min distance between two point center
    :param fb_threshold: foreground vs background threshold
    :param min_fore_count: The minimum count of foreground pixel
    :param max_fore_count: The maximum count of foreground pixel
    :param avg_fore_score: The minimum average fore score of foreground
    :param distance_weights: distance value's weights of vertical and horizontal
    :return: A list of point
    """
    c_kernel = circle_kernel(max_diameter, dtype=np.float)

    seg_smt = cv2.blur(seg, ksize=(3, 3))
    seg_smooth = cv2.blur(seg, ksize=(max_diameter + 1, max_diameter + 1))
    nms_msk = nms_mask(seg_smooth, ksize=5, dtype=np.bool)

    seg_frt = np.greater(seg, fb_threshold).astype(c_kernel.dtype)
    count = cv2.filter2D(seg_frt, -1, c_kernel)

    sel = np.logical_and(nms_msk,
                         np.greater_equal(count, min_fore_count))
    sel = np.logical_and(sel,
                         np.greater_equal(seg_smt, avg_fore_score))
    if max_fore_count > 0:
        assert max_fore_count > min_fore_count, "max_fore_count(%s) must be bigger " \
                                                "than min_fore_count(%s)" \
                                                % (max_fore_count, min_fore_count)
        sel = np.logical_and(sel, np.less_equal(count, max_fore_count))

    scr = seg_smt[sel]
    idc = ndarray_index(seg.shape)
    idc = idc[sel]

    candidates = zip(scr, idc)
    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
    points = np.zeros([0, 2], dtype=idc.dtype)
    for s, p in candidates:
        dis = points_distance(p, points, weights=distance_weights)[0]
        if np.any(dis < min_distance):
            continue
        points = np.concatenate([points, [p]], axis=0)

    return points


def seg2line(seg, fb_threshold=0.5, smooth_width=3, partition_width=20,
             partition_height=30):
    """ Find all valid lines for segmentation

    :param seg: point object segmentation result, probability map between [0.0, 1.0]
    :param fb_threshold: foreground vs background threshold
    :param smooth_width: bin width for line x-coordinate smooth
    :param partition_width: line will split when x-coordinate interval greater than partition_width
    :param partition_height: line will split when y-coordinate interval greater than partition_height
    :return: a list of lines
    """
    mask = np.greater(seg, fb_threshold)
    mask = mask.astype(np.int32)

    # solve vertical center
    height, width = mask.shape
    y_label = np.arange(height)
    y_label = np.expand_dims(y_label, axis=1)
    mask_label = np.sum(mask * y_label, axis=0)
    mask_count = np.sum(mask, axis=0)

    centers = []
    for i in range(width):
        st = max(0, i - smooth_width // 2)
        ed = min(width, i + (smooth_width + 1) // 2)
        if mask_count[i] <= 0:
            continue
        center_y = mask_label[st:ed].sum() / mask_count[st:ed].sum()
        centers.append((i, center_y))

    # partition lines
    if len(centers) == 0:
        return []

    lines = [[centers[0]]]
    for i in range(1, len(centers)):
        cp, cc = centers[i - 1], centers[i]
        # judge two point is near enough, connect near points
        if cc[0] - cp[0] < partition_width and abs(cc[1] - cp[1]) < partition_height:
            if cc[0] - cp[0] > 1:
                inter_v = np.linspace(cp[1], cc[1], cc[0] - cp[0] + 1)
                for k, v in enumerate(inter_v[1:-1]):
                    lines[-1].append((cp[0] + 1 + k, v))
        else:
            lines.append([])
        lines[-1].append(centers[i])

    return lines
