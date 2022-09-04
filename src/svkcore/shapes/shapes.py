# -*- encoding: utf-8 -*-

# @File    : shapes.py
# @Time    : 2019-11-20
# @Author  : zjh

r"""
Common shapes for object detection
"""

__all__ = ["Point", "Points", "Line", "Polygon", "Box", "Boxes", "Mask"]

import numpy as np
import cv2
import re


class _ORDER:
    width_height = 0
    height_width = 1


def _check_shape_compatible(template, shape):
    exp_shape = [x.strip() for x in template.strip("()").split(",")]
    if len(exp_shape) != len(shape):
        return False
    for e, s in zip(exp_shape, shape):
        if e == "?":
            continue
        elif e.startswith(">") or e.startswith("<"):
            if not eval(str(s) + e):
                return False
        else:
            if not (int(e) == s):
                return False
    return True


def _make_dim_compatible(template, obj, force=False):
    exp_shape = [x.strip() for x in template.strip("()").split(",")]
    shape = [int(x) if re.match(r"^[1-9]\d*$", x) else 0 for x in exp_shape]
    if np.product(shape) == 0 or force:
        obj = np.reshape(obj, shape)
    return obj


class Shape:
    """
        Base class for object detection shapes. Wrap a numpy.array object and
    add specified operations for each type shape.
    """

    order = _ORDER.width_height
    _shape = ""

    def __init__(self, obj, dtype=None):
        obj = np.asarray(obj, dtype=dtype)
        # set expect dim for empty object
        if len(obj) == 0:
            obj = _make_dim_compatible(self._shape, obj)
        if not _check_shape_compatible(self._shape, obj.shape):
            raise ValueError("Expect array shape %s, but get %s" % (self._shape, obj.shape))
        self._obj = obj

    def __getattr__(self, item):
        return getattr(self._obj, item)

    def __repr__(self):
        return "{}({!r})".format(self.__class__.__name__, self._obj)

    def __len__(self):
        return len(self._obj)

    def __getitem__(self, *args, **kwargs):
        return self._obj.__getitem__(*args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        self._obj.__setitem__(*args, **kwargs)

    def __delitem__(self, *args, **kwargs):
        self._obj.__delitem__(*args, **kwargs)

    def __add__(self, other):
        return self._obj.__add__(other)

    def __sub__(self, other):
        return self._obj.__sub__(other)

    def __mul__(self, other):
        return self._obj.__mul__(other)

    def __truediv__(self, other):
        return self._obj.__truediv__(other)

    def __mod__(self, other):
        return self._obj.__mod__(other)

    def numpy(self):
        return self._obj

    def swap(self):
        shape = self._obj.shape
        obj = np.reshape(self._obj, [-1, 2])
        obj = obj[..., ::-1]
        self._obj = np.reshape(obj, shape)
        return self


class Point(Shape):
    """Point of 2d"""

    _shape = "(2)"


class Points(Point):
    """Collection of Point"""

    _shape = "(?, 2)"

    def bounding_box(self):
        if len(self._obj) == 0:
            raise ValueError("Empty points have no bounding box.")
        top_left = np.min(self._obj, axis=0)
        bot_right = np.max(self._obj, axis=0)
        return Box(np.concatenate([top_left, bot_right]))


class Line(Points):
    """Line of 2d"""

    def length(self):
        if len(self._obj) < 2:
            return 0.
        dis = self._obj[:-1] - self._obj[1:]
        lth = np.sqrt(np.sum(dis ** 2, axis=-1)).sum()
        return lth


class Box(Shape):
    """
        Box of 2d. Record top_left and bottom_right corner position of box.
    """

    _shape = "(4)"

    def area(self):
        return (self._obj[0] - self._obj[2]) * (self._obj[1] - self._obj[3])

    def to_polygon(self):
        top_left = self._obj[:2]
        top_right = [self._obj[0], self._obj[3]]
        bot_left = [self._obj[2], self._obj[1]]
        bot_right = self._obj[2:]
        return Polygon([top_left, top_right, bot_right, bot_left])

    def to_mask(self, size=None):
        return self.to_polygon().to_mask(size)

    def bsize(self):
        """ Size of box """
        return self._obj[2:] - self._obj[:2]

    def center(self):
        return Point([(self._obj[2] + self._obj[0]) / 2,
                      (self._obj[3] + self._obj[1]) / 2])

    def to_cxywh(self):
        """ Convert box format to [center-x, center-y, width, height] """
        return np.asarray(tuple(self.center()) + tuple(self.bsize()))

    @classmethod
    def from_cxywh(cls, cxywh):
        """ Create box from format [center-x, center-y, width, height] """
        cx, cy, w, h = cxywh
        tx, ty = cx - w / 2, cy - h / 2
        bx, by = cx + w / 2, cy + h / 2
        return cls([tx, ty, bx, by])

    def scale(self, scale):
        cxywh = self.to_cxywh()
        cxywh[2:] *= scale
        return self.from_cxywh(cxywh)


class Boxes(Shape):
    """Collection of Box"""

    _shape = "(?, 4)"

    def areas(self):
        return (self._obj[:, 0] - self._obj[:, 2]) * (self._obj[:, 1] - self._obj[:, 3])

    def bsize(self):
        """ Size of boxes """
        return self._obj[:, 2:] - self._obj[:, :2]

    def center(self):
        return Points((self._obj[:, :2] + self._obj[:, 2:]) / 2)

    def to_cxywh(self):
        """ Convert box format to [center-x, center-y, width, height] """
        return np.concatenate([self.center(), self.bsize()], axis=1)

    @classmethod
    def from_cxywh(cls, cxywh):
        """ Create box from format [center-x, center-y, width, height] """
        tx, ty = cxywh[:, 0] - cxywh[:, 2] / 2, cxywh[:, 1] - cxywh[:, 3] / 2
        bx, by = cxywh[:, 0] + cxywh[:, 2] / 2, cxywh[:, 1] + cxywh[:, 3] / 2
        return cls(np.stack([tx, ty, bx, by], axis=1))

    def scale(self, scale):
        cxywh = self.to_cxywh()
        cxywh[:, 2:] *= scale
        return self.from_cxywh(cxywh)


class Polygon(Shape):
    """Polygon of 2d"""

    _shape = "(>=3, 2)"

    def area(self):
        top_left = np.min(self._obj, axis=0)
        mask = Polygon(self._obj - top_left).to_mask()
        return mask.area()

    def bounding_box(self):
        return Points(self._obj).bounding_box()

    def to_mask(self, size=None):
        if size is None:
            bot_right = np.max(self._obj, axis=0)
            size = [int(np.ceil(x)) for x in bot_right]
            if self.order == _ORDER.width_height:
                size = list(reversed(size))
        else:
            if len(size) != 2:
                raise ValueError("mask size must be tuple/list like (height, width)")
        mask = np.zeros(size, dtype=np.uint8)
        if self.order == _ORDER.width_height:
            mask = cv2.fillPoly(mask, [self._obj], (255,))
        else:
            mask = cv2.fillPoly(mask, [self._obj[:, ::-1]], (255,))
        return Mask(mask)


class Mask(Shape):
    """Mask of 2d"""

    _shape = "(?, ?)"

    def area(self):
        return np.not_equal(self._obj, 0).sum()

    def bounding_box(self):
        idx0, idx1 = np.where(self._obj)
        if len(idx0) == 0:
            return Box([0, 0, 0, 0])
        if self.order == _ORDER.width_height:
            points = np.stack([idx1, idx0], axis=-1)
        else:
            points = np.stack([idx0, idx1], axis=-1)
        return Points(points).bounding_box()

    def swap(self):
        self._obj = self._obj.T
        return self
