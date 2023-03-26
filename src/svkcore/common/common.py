# -*- encoding: utf-8 -*-

# @File    : common.py
# @Time    : 19-10-14
# @Author  : zjh

r"""
This module provides common utility functions for simple vision kit. 
It includes functions for computing md5 hash, aligning file paths, 
grouping lists, finding files recursively, collecting examples, 
and encoding/decoding images using base64.
"""

__all__ = ("str_md5", "image_md5", "get_default_font", "group",
           "basename_head", "bsn_head", "align_paths", "ifind_file_recursive",
           "find_file_recursive", "collect_examples", "collect_pascal_data",
           "group_map", "find_files", "DEFAULT_FONT_PATH",
           "b64encode_image", "b64decode_image")

import os
import hashlib
from io import BytesIO
import pkg_resources
from collections import defaultdict
from typing import Callable, Dict
import base64

from PIL import ImageFont, Image

DEFAULT_FONT_PATH = pkg_resources.resource_filename("svkcore", 'assets/DFPKingGothicGB-Light-2.ttf')


def str_md5(_bytes: bytes) -> str:
    """ Compute string md5 value

    :param _bytes: data in bytes format to be hashed by md5
    :type _bytes: bytes
    :return: md5 hash string
    :rtype: str
    """
    md5 = hashlib.md5()
    md5.update(_bytes)
    return md5.hexdigest()


def image_md5(image: Image.Image) -> str:
    """Compute a image md5 value

    :param image: pillow image object to be hashed by md5
    :type image: PIL.Image.Image
    :returns: md5 hash string
    :rtype: str
    """
    bio = BytesIO()
    image.save(bio, "JPEG")
    val = bio.getvalue()
    name_hd = str_md5(val)
    del bio
    return name_hd


def get_default_font(size: int = 24) -> ImageFont.ImageFont:
    """
    Get a default PIL.ImageFont.ImageFont instance for show label name on image
    which could deal with both English and Chinese

    :param size: font size
    :type size: int
    :return: a font object
    :rtype: PIL.ImageFont.ImageFont
    """
    return ImageFont.truetype(DEFAULT_FONT_PATH, size)


def group(lst, key, value=None) -> Dict[object, list]:
    """
    Group list by key and return a dict. Each value of the result dict is a list.
    And each list contains all values with same key.

    :param lst: a list of objects to be group
    :type lst: list
    :param key: a list of key objects or a callable function map each item in
                lst to its key
    :type key: Union[list, callable]
    :param value: a list of value objects or a callable function map each item in
                  lst to its value. Default is `None` and group will use `lst` as `value`.
    :type value: Union[list, callable]
    :return: grouped result
    :rtype: Dict[object, list]
    """
    if callable(key):
        key = map(key, lst)
    if callable(value):
        lst = map(value, lst)
    res = defaultdict(list)
    for v, k in zip(lst, key):
        res[k].append(v)
    return res


def group_map(_group: dict, func: Callable, with_key: bool = False):
    """
    Do map on a group result

    :param _group: a result dict of ``group``
    :type _group: Dict[object, list]
    :param func: a function used to process value or key and value
    :type func: Callable
    :param with_key: ``func`` process with key and value or only value.
                     Default is `False`, only process value.
    :type with_key: bool
    :return: map result dict
    :rtype: dict
    """
    if with_key:
        return {k: func(k, v) for k, v in _group.items()}
    return {k: func(v) for k, v in _group.items()}


def basename_head(path, sep=".", align_left=False):
    """
    Get basename head of a path

    :param path: a path
    :type path: str
    :param sep: a separator str to split path's basename. Default is ``"."``.
    :type sep: str
    :param align_left: split basename with separator from left or not. Default is False.
                       And this function will split basename from right.
    :type align_left: bool
    :return: the head part of basename of ``path``
    :rtype: str
    """
    if align_left:
        return os.path.basename(path).split(sep, 1)[0]
    else:
        return os.path.basename(path).rsplit(sep, 1)[0]


def align_paths(paths0, paths1, *args, sort=False, key_fn=None):
    """
    Align paths base on its name head
    This function will delete dis-matched paths with no prompt.

    :param paths0: the first list of file paths
    :type paths0: list
    :param paths1: the second list of file paths
    :type paths1: list
    :param args: the others list of file paths
    :param sort: sorted output by key
    :type sort: bool
    :param key_fn: extract align key function, default is :py:func:`basename_head`
    :type key_fn: callable
    :return: align paths in list
    :rtype: list
    """
    if key_fn is None:
        key_fn = basename_head
    assert callable(key_fn), "key_fn must be callable"

    paths_list = [paths0, paths1] + list(args)
    heads_list = [[key_fn(p) for p in paths]
                  for paths in paths_list]
    head_path_map_list = [dict(zip(heads, paths))
                          for heads, paths in zip(heads_list, paths_list)]

    valid_heads = set(heads_list[0])
    for hs in heads_list[1:]:
        valid_heads = valid_heads.intersection(hs)
    if sort:
        valid_heads = sorted(valid_heads)

    aligned_paths_list = [[mp[h] for h in valid_heads]
                          for mp in head_path_map_list]

    return aligned_paths_list


def ifind_file_recursive(directory, suffixes, ignore_case=False):
    """
    Find all files with provided suffixes

    :param directory: the target directory
    :type directory: str
    :param suffixes: a suffix or a list of suffixes of file to be find
    :type suffixes: Union[str, list]
    :param ignore_case: match file suffix in case ignore mode
    :type ignore_case: bool
    :return: a generator which returns the path to the file which meets
             the postfix
    :rtype: a generator
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError("[Errno 2] No such file or directory: '%s'" % directory)

    if isinstance(suffixes, str):
        suffixes = [suffixes]

    def _suffix_match(filename):
        if ignore_case:
            return any([filename.lower().endswith(x.lower()) for x in suffixes])
        else:
            return any([filename.endswith(x) for x in suffixes])

    for root, dirs, files in os.walk(directory):
        for file in files:
            if _suffix_match(file):
                yield os.path.join(root, file)


def find_file_recursive(directory, suffixes, ignore_case=False):
    """
    Find all files with provided suffixes

    :param directory: the target directory
    :type directory: str
    :param suffixes: a suffix or a list of suffixes of file to be find
    :type suffixes: Union[str, list]
    :param ignore_case: match file suffix in case ignore mode
    :type ignore_case: bool
    :return: a list which returns the path to the file which meets
             the postfix
    :rtype: list
    """
    return list(ifind_file_recursive(directory, suffixes, ignore_case=ignore_case))


ifind_files = ifind_file_recursive
find_files = find_file_recursive


def collect_examples(directory, suffixes_list, ignore_case=False,
                     sort=False, key_fn=None):
    """
    Collect examples from one directory base on given suffixes list

    :param directory: Root directory of examples
    :type directory: str
    :param suffixes_list: A list of suffix list for example's each part
    :type suffixes_list: list
    :param ignore_case: ignore case when match file name suffix. Default is ``False``.
    :type ignore_case: bool
    :param sort: sort the result list by key. Default is ``False``.
    :type sort: bool
    :param key_fn: a function use to extract align key, default is :py:func:`basename_head`
    :type key_fn: callable
    :return: all matched file paths list
    :rtype: list
    """
    paths_list = [find_file_recursive(directory, suffixes, ignore_case=ignore_case)
                  for suffixes in suffixes_list]
    aligned_paths_list = align_paths(*paths_list, sort=sort, key_fn=key_fn)
    examples = list(zip(*aligned_paths_list))
    return examples


def collect_pascal_data(directory):
    """
    Collect pascal format dataset

    :param directory: the root directory of a pascal dataset
    :type directory: str
    :return: two list of images and annotations of the the pascal dataset
    :rtype: list
    """
    examples = collect_examples(directory, [(".jpg", ".jpeg", ".png", ".bmp"), ".xml"],
                                ignore_case=True)
    return examples


bsn_head = basename_head


def b64encode_image(image: Image.Image, format: str = "JPEG") -> bytes:
    """
    Convert PIL.Image.Image object to bytes data use base64 encode

    :param image: an instance of PIL.Image.Image
    :type image: PIL.Image.Image
    :param format: a string represents image encoding format. could be "JPEG" or "PNG"
    :type format: str
    :return: base64 encoded image data
    :rtype: bytes
    """
    bio = BytesIO()
    image.save(bio, format)
    enc_dt = base64.b64encode(bio.getvalue())
    del bio
    return enc_dt


def b64decode_image(data: bytes) -> Image.Image:
    """
    Decode bytes data of encoded image to an instance of PIL.Image.Image

    :param data: base64 encoded bytes image data
    :type data: bytes
    :return: An instance of Image.Image represents the decode image
    :rtype: PIL.Image.Image
    """
    enc_dt = base64.b64decode(data)
    bio = BytesIO(enc_dt)
    image = Image.open(bio)
    del bio
    return image
