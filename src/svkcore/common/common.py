# -*- encoding: utf-8 -*-

# @File    : common.py
# @Time    : 19-10-14
# @Author  : zjh

r"""
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
from typing import Callable
import base64

from PIL import ImageFont, Image

DEFAULT_FONT_PATH = pkg_resources.resource_filename("svkcore", 'assets/DFPKingGothicGB-Light-2.ttf')


def str_md5(bytes_):
    """Compute string md5 value"""
    md5 = hashlib.md5()
    md5.update(bytes_)
    return md5.hexdigest()


def image_md5(image):
    """Compute a image md5 value"""
    bio = BytesIO()
    image.save(bio, "JPEG")
    val = bio.getvalue()
    name_hd = str_md5(val)
    del bio
    return name_hd


def get_default_font(size=24):
    """get a default PIL.ImageFont instance for show label name on image
    which could deal with both English and Chinese

    Args:
        size(int): font size
    Returns:
        PIL.ImageFont: a ImageFont instance
    """
    return ImageFont.truetype(DEFAULT_FONT_PATH, size)


def group(lst, key, value=None) -> dict:
    """Group list by key"""
    if callable(key):
        key = map(key, lst)
    if callable(value):
        lst = map(value, lst)
    res = defaultdict(list)
    for v, k in zip(lst, key):
        res[k].append(v)
    return res


def group_map(_group: dict, func: Callable, with_key=False):
    """Do map on a group result"""
    if with_key:
        return {k: func(k, v) for k, v in _group.items()}
    return {k: func(v) for k, v in _group.items()}


def basename_head(path, sep=".", align_left=False):
    """Get basename head of a path """
    if align_left:
        return os.path.basename(path).split(sep, 1)[0]
    else:
        return os.path.basename(path).rsplit(sep, 1)[0]


def align_paths(paths0, paths1, *args, sort=False, key_fn=None):
    """ Align paths base on its name head
        This function will delete dis-matched paths with no promot.
        Args:
            paths0: the first of list file paths
            paths1: the second of list file paths
            args: others of list file paths
            sort: sorted output by key
            key_fn: extract align key function, default is `basename_head`
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
    """ Find all file with set postfixes

    :param directory: the target directory
    :param suffixes: a suffix or a list of suffixes of file to be find
    :param ignore_case: match file suffix in case angnose mode
    :return: a generator which returns the path to the file which meets
        the postfix
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
    """ Find all file with set postfixes

    :param directory: the target directory
    :param suffixes: a suffix or a list of suffixes of file to be find
    :param ignore_case: match file suffix in case angnose mode
    :return: a list which returns the path to the file which meets
        the postfix
    """
    return list(ifind_file_recursive(directory, suffixes, ignore_case=ignore_case))


ifind_files = ifind_file_recursive
find_files = find_file_recursive


def collect_examples(directory, suffixes_list, ignore_case=False,
                     sort=False, key_fn=None):
    """ Collect examples

        Collect examples from one directory base on given suffixes list

    :param directory: Root directory of examples
    :param suffixes_list: A list of suffix list for example's each part
    :param ignore_case:
    :param sort:
    :param key_fn:
    :return:
    """
    paths_list = [find_file_recursive(directory, suffixes, ignore_case=ignore_case)
                  for suffixes in suffixes_list]
    aligned_paths_list = align_paths(*paths_list, sort=sort, key_fn=key_fn)
    examples = list(zip(*aligned_paths_list))
    return examples


def collect_pascal_data(directory):
    """ Collect pascal format dataset """
    examples = collect_examples(directory, [(".jpg", ".jpeg", ".png", ".bmp"), ".xml"],
                                ignore_case=True)
    return examples


bsn_head = basename_head


def b64encode_image(image: Image.Image, format: str="JPEG") -> bytes:
    """
    Convert PIL.Image.Image object to bytes data use base64 encode
    :param image: an instance of PIL.Image.Image
    :param format: a string represents image encoding format.
        could be "JPEG" or "PNG"
    :return: base64 encoded image data
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
    :return: An instance of Image.Image represents the decode image
    """
    enc_dt = base64.b64decode(data)
    bio = BytesIO(enc_dt)
    image = Image.open(bio)
    del bio
    return image
