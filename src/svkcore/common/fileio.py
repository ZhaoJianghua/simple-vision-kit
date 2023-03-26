# -*- encoding: utf-8 -*-

# @File    : fileio.py
# @Time    : 19-10-14
# @Author  : zjh

r"""
This module provides functions for file input/output operations,
including copying files, loading and saving json, pickle,
csv files, and reading and writing images using OpenCV.
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
    """
    Copy files in list ``paths`` from ``src_dir`` to ``dst_dir``

    :param paths: a list of paths
    :type paths: list
    :param dst_dir: destination directory
    :type dst_dir: str
    :param src_dir: root directory of all paths
    :type src_dir: str
    """
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
    """
    Load json format file

    :param path: json file path
    :type path: str
    :param encoding: json file encoding like ``utf-8``, etc.
    :type encoding: str
    :param kwargs: other options for `json.load`
    :return: loaded json object
    :type: object
    """
    with open(path, encoding=encoding) as f:
        return json.load(f, **kwargs)


def save_json(obj, path, indent=2, ensure_ascii=False, *args, **kwargs):
    """
    Save object as json format

    :param obj: object to be saved as json format
    :type obj: object
    :param path: json file path
    :type path: str
    :param indent: number of spaces for indentation
    :type indent: int
    :param ensure_ascii: whether to ensure ascii encoding
    :type ensure_ascii: bool
    :param args: other options for `json.dump`
    :param kwargs: other options for `json.dump`
    """
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=ensure_ascii,
                  *args, **kwargs)


def load_pickle(path):
    """
    Load object from pickle format

    :param path: pickle file path
    :type path: str
    :return: loaded object
    :rtype: object
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path):
    """
    Save object as pickle format

    :param obj: object to be saved as pickle format
    :type obj: object
    :param path: pickle file path
    :type path: str
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def cv2imread(path):
    """
    Read image from file path using OpenCV

    :param path: image file path
    :type path: str
    :return: loaded image as numpy array
    :rtype: numpy.ndarray
    """
    with open(path, "rb") as f:
        img_np = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(img_np, cv2.IMREAD_COLOR)


def cv2imwrite(path, img):
    """
    This function is a compatible version of `cv2.imwrite`. It writes
    an image to the specified file path in .jpg format.

    :param path: output image file path
    :type path: str
    :param img: output image data
    :type img: numpy.ndarray
    """
    with open(path, 'wb') as f:
        _, buffer = cv2.imencode(".jpg", img)
        f.write(buffer)


def load_csv(path, *, with_header=True, encoding=None):
    """
    This function is used to load csv format file.
    It reads the csv file from the given path and returns the
    rows as a list of lists. If with_header is True, the first
    row is considered as the header and returned separately.
    If encoding is provided, it is used to decode the csv file.

    :param path: csv file path
    :type path: str
    :param with_header: return result with data header
    :type with_header: bool
    :param encoding: csv file encoding like ``utf-8``, etc.
    :type encoding: str
    :return:
    :rtype: list
    """
    with open(path, encoding=encoding) as f:
        rows_ = map(lambda row: [item.strip() for item in row], csv.reader(f))
        if with_header:
            head, *rows = rows_
            return head, rows
        else:
            return list(rows_)


def save_csv(rows, path, *, header=None, encoding=None):
    """
    This function is used to save data into csv format file.
    If header is provided, header will place at the first row of the output csv file.
    If encoding is provided, it is used to decode the csv file.

    :param rows: a list of data row
    :type rows: list
    :param path: csv file path
    :type path: str
    :param header: data header
    :type header: list
    :param encoding: csv file encoding like ``utf-8``, etc.
    :type encoding: str
    """
    with open(path, "w", encoding=encoding) as f:
        if header:
            rows = [header] + rows
        csv.writer(f).writerows(rows)
