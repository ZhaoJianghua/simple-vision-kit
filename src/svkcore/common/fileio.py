# -*- encoding: utf-8 -*-

# @File    : fileio.py
# @Time    : 19-10-14
# @Author  : zjh

r"""
"""

__all__ = ("copy_files", "load_json", "save_json", "load_pickle", "save_pickle",
           "cv2imread", "cv2imwrite", "load_csv", "save_csv")

import shutil
import os
import json
import csv

import pickle
import cv2
import numpy as np


def copy_files(paths, dst_dir, src_dir=None):
    """ Copy files """
    dst_dir = dst_dir.rstrip(os.path.pathsep)
    if src_dir:
        src_dir = src_dir.rstrip(os.path.pathsep)

    os.makedirs(dst_dir, exist_ok=True)
    for path in paths:
        if src_dir is not None:
            dst_path = path.replace(src_dir, dst_dir)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        else:
            dst_path = os.path.join(dst_dir, os.path.basename(path))
        shutil.copyfile(path, dst_path)


def load_json(path, *, encoding=None, **kwargs):
    """Load json format file"""
    with open(path, encoding=encoding) as f:
        return json.load(f, **kwargs)


def save_json(obj, path, indent=2, ensure_ascii=False, *args, **kwargs):
    """Save object as json format"""
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=ensure_ascii,
                  *args, **kwargs)


def load_pickle(path):
    """ Load pickle """
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path):
    """ Save object as pickle format """
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def cv2imread(path):
    """ Compatible cv2.imread """
    with open(path, "rb") as f:
        img_np = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(img_np, cv2.IMREAD_COLOR)


def cv2imwrite(path, img):
    """ Compatible cv2.imwrite """
    with open(path, 'wb') as f:
        _, buffer = cv2.imencode(".jpg", img)
        f.write(buffer)


def load_csv(path, *, with_header=True, encoding=None):
    """Load csv format file"""
    with open(path, encoding=encoding) as f:
        rows_ = map(lambda row: [item.strip() for item in row], csv.reader(f))
        if with_header:
            head, *rows = rows_
            return head, rows
        else:
            return list(rows_)


def save_csv(rows, path, *, header=None, encoding=None):
    """Save rows as csv format"""
    with open(path, "w", encoding=encoding) as f:
        if header:
            rows = [header] + rows
        csv.writer(f).writerows(rows)
