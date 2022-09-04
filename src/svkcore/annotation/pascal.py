# -*- encoding: utf-8 -*-

# @File    : pascal.py
# @Time    : 20-02-10
# @Author  : zjh

r"""
"""

__all__ = ("read_annotation", "write_annotation", "DTObject", "DTAnnotation",
           "DTDataset")

import os
import json
from typing import List
from collections import defaultdict
import shutil

from lxml import etree
import numpy as np

from svkcore.shapes import *
from svkcore.common import *


def _recursive_parse_xml_to_dict(xml):
    """Recursively parses XML contents to python dict.

    We assume that `object` and `point` tags are the only twos that can appear
    multiple times at the same level of a tree.

    Args:
      xml: xml tree obtained by parsing XML file contents using lxml.etree

    Returns:
      Python dictionary holding XML contents.
    """
    if xml is None:
        return {}
    if len(xml) == 0:
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = _recursive_parse_xml_to_dict(child)
        if child.tag not in ('object', 'point'):
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def _recursive_create_dict_to_xml(dct, root):
    """Recursively create XML contents base on a python dict.

    Args:
      dct: python dictionary holding XML contents
      root: xml tree root where dict contents will append on.

    Returns:
    """
    for key, val in dct.items():
        if isinstance(val, dict):
            node = etree.SubElement(root, key)
            _recursive_create_dict_to_xml(val, node)
        elif isinstance(val, list):
            for x in val:
                node = etree.SubElement(root, key)
                _recursive_create_dict_to_xml(x, node)
        else:
            node = etree.SubElement(root, key)
            node.text = str(val)


def read_annotation(annotation_path):
    """ Read object detection annotation of xml format file
    
    Args:
      annotation_path: file path of annotation

    Returns: a dict of detection annotation
    """
    with open(annotation_path, 'rb') as f:
        xml_str = f.read()
    xml = etree.fromstring(xml_str)
    data = _recursive_parse_xml_to_dict(xml)["annotation"]
    return data


def write_annotation(annotation_path, annotation):
    """ Write object detection annotation to a xml format file
    
    Args:
      annotation_path: file path of annotation
      annotation: a dict of detection annotation

    Returns: 
    """
    root = etree.Element("annotation")
    _recursive_create_dict_to_xml(annotation, root)  # write to file:
    tree = etree.ElementTree(root)
    tree.write(annotation_path, pretty_print=True, encoding='utf-8')


class DTObject(object):
    """ Detection object: base object for object detection """

    box_keys = ("xmin", "ymin", "xmax", "ymax")

    def __init__(self, name, bndbox=None, polygon=None, mask=None,
                 pose="Unspecified", truncated=False, difficult=False):
        assert bndbox is not None or polygon is not None, "bndbox/polygon is both None!"
        self.name = name
        self.bndbox = Box(bndbox) if bndbox is not None else None
        self.polygon = Polygon(polygon) if polygon is not None else None
        self.mask = mask  # base on bndbox
        self.pose = pose
        self.truncated = bool(truncated)
        self.difficult = bool(difficult)

    @staticmethod
    def loadd(obj: dict):
        bndbox, polygon = None, None
        if 'bndbox' in obj:
            bndbox = [int(round(float(obj['bndbox'][k]))) for k in DTObject.box_keys]
        elif 'polygon' in obj:
            polygon = [(int(round(float(p['x']))), int(round(float(p['y']))))
                       for p in obj['polygon']['point']]
        else:
            raise ValueError("Need bndbox or polygon!")

        return DTObject(obj['name'], bndbox, polygon, None,
                        obj.get('pose', 'Unspecified'),
                        bool(int(obj.get('truncated', 0))),
                        bool(int(obj.get('difficult', 0))))

    def dumpd(self) -> dict:
        if self.bndbox is not None:
            bndbox = {k: int(v) for k, v in zip(DTObject.box_keys, self.bndbox)}
            key = "bndbox"
        elif self.polygon is not None:
            bndbox = {"point": [{"x": int(p[0]), "y": int(p[1])} for p in self.polygon]}
            key = "polygon"
        else:
            raise ValueError("Need bndbox or polygon!")

        return {"name": self.name,
                key: bndbox,
                "pose": self.pose,
                "truncated": int(self.truncated),
                "difficult": int(self.difficult)}

    def __str__(self):
        return json.dumps(self.dumpd(), indent=2, ensure_ascii=False)


class DTAnnotation(object):
    """ Detection Annotation: An annotation for object detection """

    size_keys = ('width', 'height', 'depth')

    def __init__(self, filename: str, size, objects: List[DTObject], segmented=False, **kwargs):
        self.filename = filename
        self.size = size
        self.segmented = bool(segmented)
        self.objects = objects
        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def loadd(obj: dict):
        filename = obj.get('filename')
        size = obj.get('size', {})
        size = tuple(int(size[k]) for k in DTAnnotation.size_keys if k in size)
        objects = [DTObject.loadd(x) for x in obj.get('object', [])]
        return DTAnnotation(filename, size, objects)

    @staticmethod
    def load(path: str):
        return DTAnnotation.loadd(read_annotation(path))

    def dumpd(self):
        size = dict(zip(DTAnnotation.size_keys, self.size))
        objects = [DTObject.dumpd(x) for x in self.objects]
        return {'filename': self.filename,
                'size': size,
                'segmented': int(self.segmented),
                'object': objects}

    def dump(self, path):
        write_annotation(path, self.dumpd())

    def __str__(self):
        return json.dumps(self.dumpd(), indent=2, ensure_ascii=False)

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, item) -> DTObject:
        return self.objects[item]


class DTDataset(object):
    """ Detection dataset: A collection of annotations for object detection """

    def __init__(self, annotations, images, categories):
        """
        Init dataset
        :param annotations: A list of DTAnnotation
        :param annotations: A list of image path
        :param categories: A list of {"id": 1, "name": "xxx", "supercategory": "none"}, id in [1, #categories]
        """
        self.annotations = annotations
        self.images = images
        self.categories = categories

    @staticmethod
    def load_coco(annotation_path, image_root):
        """
        Load coco format dataset
        :param annotation_path: json format annotation file path
        :param image_root: image root directory
        :return: DTDataset object
        """
        dataset = load_json(annotation_path)
        img2anns = defaultdict(list)
        for ann in dataset['annotations']:
            img2anns[ann['image_id']].append(ann)
        categories = dataset['categories']
        id2cls = {x['id']: x['name'] for x in categories}
        annotations, images = [], []
        for img in dataset['images']:
            images.append(os.path.join(image_root, img['file_name']))
            size = tuple(img[x] for x in ("width", "height"))
            objects = []
            for _ann in img2anns[img["id"]]:
                bbox = _ann['bbox']
                bbox = bbox[:2] + [bbox[0] + bbox[2], bbox[1] + bbox[3]]
                objects.append(DTObject(id2cls[_ann['category_id']], bbox))
            ann = DTAnnotation(img['file_name'], size, objects)
            annotations.append(ann)
        return DTDataset(annotations, images, categories)

    @staticmethod
    def load_pascal(annotation_paths, image_paths):
        """
        Load pascal format dataset
        :param annotation_paths: a list of pascal format annotation file path
        :param image_paths: a list of image path respect with each annotation file
        :return: DTDataset object
        """
        assert len(annotation_paths) == len(image_paths), \
            "The length of annotation paths and image paths should be same!"
        annotations = []
        categories = []
        names = set()
        for anp, imp in zip(annotation_paths, image_paths):
            ann = DTAnnotation.load(anp)
            ann.filename = imp
            for obj in ann.objects:
                if obj.name not in names:
                    categories.append({"id": len(categories) + 1,
                                       "name": obj.name,
                                       "supercategory": "none"})
                    names.add(obj.name)
            annotations.append(ann)
        return DTDataset(annotations, images=image_paths, categories=categories)

    def dump_coco(self, path):
        """
        Save dataset to coco format
        :param path: coco format annotation path
        :return: None
        """
        annotations = []
        images = []
        name2catid = {cat['name']: cat['id'] for cat in self.categories}
        img_id = 0
        ann_id = 0
        for i, ann in enumerate(self.annotations):
            img_id += 1
            file_name = self.images[i] if self.images else ann.filename
            images.append({"id": img_id,
                           "file_name": os.path.basename(file_name),
                           "width": ann.size[0],
                           "height": ann.size[1]})
            for obj in ann.objects:
                ann_id += 1
                box = obj.bndbox.tolist()
                annotations.append({"id": ann_id,
                                    "category_id": name2catid[obj.name],
                                    "image_id": img_id,
                                    "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                                    "area": Box(obj.bndbox).area().item(),
                                    "segmentation": [],
                                    "iscrowd": 0,
                                    "ignore": 0})

        with open(path, "w") as f:
            json.dump({"annotations": annotations,
                       "images": images,
                       "categories": self.categories}, f)

    def dump_pascal(self, annotation_dir):
        """
        Save dataset to pascal format
        :param annotation_dir: pascal format annotations directory
        :return: None
        """
        os.makedirs(annotation_dir, exist_ok=True)
        for ann in self.annotations:
            dp = os.path.join(annotation_dir, bsn_head(ann.filename) + ".xml")
            ann.dump(dp)

    def dump_yolo(self, dataset_dir):
        """
        Save dataset to yolo format
        :param dataset_dir: yolo format dataset directory
        :return: None
        """
        label_dir = os.path.join(dataset_dir, "labels")
        image_dir = os.path.join(dataset_dir, "images")
        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)
        cls2id = {x['name']: x['id'] - 1 for x in self.categories}
        for ann in self.annotations:
            dp = os.path.join(label_dir,  bsn_head(ann.filename) + ".txt")
            sz = ann.size[:2]
            with open(dp, "w") as f:
                for obj in ann:
                    cxywh = Box(obj.bndbox).to_cxywh() / (sz + sz)
                    item = "%s %.6f %.6f %.6f %.6f\n" % (cls2id[obj.name], *cxywh)
                    f.write(item)
            dp = os.path.join(image_dir,  os.path.basename(ann.filename))
            shutil.copyfile(ann.filename, dp)
