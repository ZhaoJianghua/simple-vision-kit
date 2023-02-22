# -*- encoding: utf-8 -*-

# @File    : visualize.py
# @Time    : 2019-11-20
# @Author  : zjh

r"""
"""

__all__ = ["draw_points", "draw_lines", "draw_boxes", "draw_polygons", "draw_masks",
           "draw_texts", "draw_boxes_texts", "draw_detection_result", "generate_colors",
           "draw_annotation", "DEFAULT_COLORS", "cv2image2pil", "pil2cv2image",
           "images_gallery"]

import colorsys
import math
from typing import List, Tuple, Union

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageColor
import cv2

from svkcore.common import get_default_font
from svkcore.shapes import *


def draw_points(image, points, color="red", scale=3, shape="t"):
    """ Draw boxes on points

    :param image: A PIL.Image object
    :param points: a list of points
    :param color: points color
    :param scale: points scale
    :param shape: visualize shape, 't' for triangle else for circle
    :return: drew image
    """
    draw = ImageDraw.Draw(image)
    pts = Points(points)
    if shape == "t":
        for pt in pts:
            pt = tuple([int(x) for x in pt[::-1]])
            top_left = (pt[0] - scale, pt[1] - 2 * scale)
            top_right = (pt[0] + scale, pt[1] - 2 * scale)
            draw.polygon([pt, top_left, top_right, pt], fill=color)
    else:
        for pt in pts:
            rect = tuple(pt - scale / 2) + tuple(pt + scale / 2)
            draw.ellipse(rect, fill=color)
    del draw
    return image


def draw_lines(image, lines, color="red", width=0):
    """ Draw boxes on lines

    :param image: A PIL.Image object
    :param lines: a list of lines
    :param color: lines color
    :param width: line width
    :return: drew image
    """
    draw = ImageDraw.Draw(image)
    for line in lines:
        line = Line(line)
        draw.line([tuple(p) for p in line], fill=color, width=width)
    del draw
    return image


def draw_boxes(image, boxes, color="red", width=0, fullfill=False):
    """ Draw boxes on image

    :param image: A PIL.Image object
    :param boxes: a list of boxes
    :param color: boxes color
    :param width: line width
    :param fullfill: full fill boxes or not
    :return: drew image
    """
    draw = ImageDraw.Draw(image)
    for box in boxes:
        polygon = Box(box).to_polygon()
        polygon = [tuple(x) for x in polygon]
        if fullfill:
            draw.polygon(polygon, fill=color)
        else:
            polygon += polygon[:1]
            draw.line(polygon, width=width, fill=color)
    del draw
    return image


def draw_polygons(image, polygons, color="red", width=0, fullfill=False):
    """ Draw polygons on image

    :param image: A PIL.Image object
    :param polygons: a list of polygons
    :param color: mask color
    :param width: line width
    :param fullfill: full fill polygon or not
    :return: drew image
    """
    draw = ImageDraw.Draw(image)
    for polygon in polygons:
        polygon = Polygon(polygon)
        polygon = [tuple(x) for x in polygon]
        if fullfill:
            draw.polygon(polygon, fill=color)
        else:
            polygon += polygon[:1]
            draw.line(polygon, width=width, fill=color)
    del draw
    return image


def draw_masks(image, masks, color='red', alpha=0.5):
    """ Draw masks on image

    :param image: A PIL.Image object
    :param masks: a list of masks
    :param color: mask color
    :param alpha: transport alpha
    :return: drew image
    """
    color = ImageColor.getcolor(color, image.mode)
    image_arr = np.array(image)
    for mask in masks:
        mask = np.not_equal(mask, 0)
        if isinstance(color, tuple):
            mask = np.expand_dims(mask, axis=-1)
        image_arr = (1 - mask * alpha) * image_arr + mask * (1 - alpha) * color
    image_arr = np.round(image_arr).astype(np.uint8)
    image.paste(Image.fromarray(image_arr))
    return image


def draw_texts(image, xys, texts, color='red', back_color=None, font_size=12,
               position="topleft", offset=(0, 0), margin=(0, 0, 0, 0)):
    """ Draw texts on image

    :param image: A PIL.Image object
    :param xys: A list of corner's coordinate where text align
    :param texts: A list of texts
    :param color: Text color
    :param back_color:
    :param font_size: Font size
    :param position: text align position, enum string like:
                     topleft/topright/bottomleft/bottomright/manu
    :param offset: A tuple vector of offset for 'manu' position
    :param margin: A tuple (left, top, right, bottom) denotes the margins to back boarder
    :return: drew image
    """
    assert position in ("topleft", "topright", "bottomleft",
                        "bottomright", "manu"), "Invalid position!"

    draw = ImageDraw.Draw(image)
    font = get_default_font(font_size)
    for xy, text in zip(xys, texts):
        text_size = draw.textsize(text, font)
        if position == "topleft":
            xy = tuple(xy)
        elif position == "topright":
            xy = (xy[0] - text_size[0], xy[1])
        elif position == "bottomleft":
            xy = (xy[0], xy[1] - text_size[1])
        elif position == "bottomright":
            xy = (xy[0] - text_size[0], xy[1] - text_size[1])
        elif position == "manu":
            xy = (xy[0] + offset[0], xy[1] + offset[1])
        else:
            raise ValueError("Invalid position!")

        if back_color:
            margin_w = margin[0] + margin[2]
            margin_h = margin[1] + margin[3]
            draw.rectangle([tuple(xy), (xy[0] + text_size[0] + margin_w,
                                        xy[1] + text_size[1] + margin_h)],
                           fill=back_color)
        xy = (xy[0] + margin[0], xy[1] + margin[1])
        draw.text(xy, text, fill=color, font=font)
    del draw
    return image


def draw_boxes_texts(image, boxes, texts, width=1, color='red'):
    """ Draw boxes and its text information on image

    :param image: A PIL.Image object
    :param boxes: A list of box
    :param texts: A list of text
    :param width: Line width, determines thickness of box and font size of text
    :param color: Color of boxes and texts
    :return: drew image
    """
    text_margin = (4, 2)
    draw_boxes(image, boxes, color, width)
    xys = [[text_margin[0] + b[0], text_margin[1] + b[1]] for b in boxes]
    draw_texts(image, xys, texts, color, font_size=int(round(width * 12)))
    return image


def generate_colors(num):
    """Generate colors for drawing bounding boxes"""
    hsv_tuples = [(x / num, 1., 1.) for x in range(num)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: tuple(int(xx * 255) for xx in x), colors))
    np.random.seed(4)  # Fixed seed for consistent colors across runs.
    np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    np.random.seed(None)  # Reset seed to default.
    return colors


def draw_detection_result(image, boxes, classes, display_strings,
                          color_nums=100, scale=-1.):
    """ Draw detection result

    :param image: A PIL.Image object
    :param boxes: A list of box
    :param classes: A list of detection class index
    :param display_strings: A list of display string
    :param color_nums: The max number of different colors
    :param scale: visualize box and text scale. Default -1.0 means
        auto adjust scale by input image
    :return: drew image
    """
    if scale <= 0.0:
        scale = min(image.size) / 400.
    colors = generate_colors(color_nums)
    for bx, cls, display_str in zip(boxes, classes, display_strings):
        color = colors[cls % color_nums]
        draw_boxes(image, [bx], color, width=int(math.ceil(scale)))
        draw_texts(image, [bx[:2]], [display_str], color="black", back_color=color,
                   font_size=int(round(12*scale)), position="bottomleft")
    return image


DEFAULT_COLORS = tuple(generate_colors(99))


def draw_annotation(image, annotation, name2cls, color_table=DEFAULT_COLORS,
                    add_unknown_name=False):
    """ Draw DTAnnotation to an image

    :param image: A PIL.Image object
    :param annotation: An instance of DTAnnotation
    :param name2cls: a dict of name to its class id number
    :param color_table: each class colors
    :param add_unknown_name: whether add a new name to name2cls, default is False
    :return: drew image
    """
    scale = min(image.size) / 400.
    box_width = int(math.ceil(scale))
    font_size = int(round(12*scale))

    for obj in annotation.objects:
        if obj.name not in name2cls:
            if add_unknown_name:
                print("add %s: %s" % (obj.name, len(name2cls)))
                name2cls[obj.name] = len(name2cls)
            else:
                raise ValueError("Unknown name! (%s)" % obj.name)

        color = color_table[name2cls[obj.name] % len(color_table)]

        if obj.bndbox is not None:
            image = draw_boxes(image, [obj.bndbox], color, width=box_width)
            image = draw_texts(image, [obj.bndbox[:2]], [obj.name], color,
                               back_color="black", position="bottomleft",
                               font_size=font_size)
        elif obj.polygon is not None:
            image = draw_polygons(image, [obj.polygon], color, width=box_width)
            poly = sorted(sorted(obj.polygon.tolist(), key=lambda x: x[0]),
                          key=lambda x: x[1])
            image = draw_texts(image, [poly[0]], [obj.name], color,
                               back_color="black", position="bottomleft",
                               font_size=font_size)

    return image


def cv2image2pil(cv2_image: np.ndarray) -> Image.Image:
    """ Convert openCV format image to PIL.Image.Image

    :param cv2_image: openCV format image instance
    :return: converted Image.Image instance
    """
    if cv2_image.ndim == 3 and cv2_image.shape[-1] == 3:
        return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
    if cv2_image.ndim == 3 and cv2_image.shape[-1] == 4:
        return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGRA2RGBA))
    if cv2_image.ndim == 2:
        return Image.fromarray(cv2_image)
    raise TypeError('cv2_image shape (%s) is invalid!' % cv2_image.shape)


def pil2cv2image(image: Image.Image) -> np.ndarray:
    """
    Convert PIL.Image.Image to openCV format image
    :param image: an instance of PIL.Image.Image
    :return: converted openCV format image
    """
    cv2_image = np.asarray(image)
    if cv2_image.ndim == 2:
        return cv2_image
    if cv2_image.ndim == 3 and cv2_image.shape[-1] == 3:
        return cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)
    if cv2_image.ndim == 3 and cv2_image.shape[-1] == 4:
        return cv2.cvtColor(cv2_image, cv2.COLOR_RGBA2BGRA)
    raise TypeError(f'image color format(f{image.mode}) is not support.')


def images_gallery(image_list: Union[Tuple[Image.Image], List[Image.Image]],
                   n_cols: int = 6,
                   n_rows: int = None,
                   cell_size: Union[Tuple[int], List[int]] = (224, 224),
                   pad: int = 16,
                   align: int = 0,
                   back_color: Union[Tuple[int], List[int], str] = "black",
                   same_scale: bool = False) -> Image.Image:
    """ Paste a list of images into one panel for better visualize

    :param image_list: A list of Image.Image instance.
    :param n_cols: Max number of images to show in each row.
           If the number of `image_list` is less than `n_cols`,
           `n_cols` will be set as the number of `image_list`.
           Default value is 6.
    :param n_rows: Number of rows to show images. If `n_cols` is set,
           `n_rows` will be automatically calculated by $ceil(\frac{N}{n_cols})$.
           Otherwise `n_cols` will be automatically calculated.
           Default value is None.
    :param cell_size: The size of cell where each image is placed in.
    :param pad: The pad width/height between two cells.
    :param align: The align mode. Set 0(center), 1(left/up), 2(right/bottom) to choose
           align mode in each cell.
    :param back_color: Background color.
    :param same_scale: A boolean value indicates whether to use a same scale factor
           to resize all images in image_list. Set it to be True if you want to visualize
           the sizes of different images.
    :return: Pasted Image.Image instance.
    """
    num = len(image_list)
    pad = max(0, pad)
    if num <= 0:
        pad = max(4, pad)
        return Image.new("RGB", (pad, pad))

    assert n_cols or n_rows, 'Set at least one value of (n_cols, n_rows)'
    if n_cols:
        n_rows = math.ceil(num / n_cols)
    else:
        n_cols = math.ceil(num / n_rows)
    if num < n_cols:
        n_cols = num

    assert any(x > 0 for x in cell_size), "Set at least on value of cell_size(w, h)"
    assert align in (0, 1, 2), "Only support 0(center), 1(left/up), 2(right/bottom) align mode."

    # convert images + paint plan
    factors = np.zeros(num)
    for i, img in enumerate(image_list):
        if isinstance(img, np.ndarray):
            img = cv2image2pil(img)
        if not isinstance(img, Image.Image):
            raise ValueError('Invalid image type %s' % type(img))
        img = img.convert('RGB')
        if all(x > 0 for x in cell_size):
            fac = min(a / b for a, b in zip(cell_size, img.size))
        elif cell_size[0] > 0:
            fac = cell_size[0] / img.size[0]
        else:
            fac = cell_size[1] / img.size[1]
        factors[i] = fac
        image_list[i] = img

    # use same scale factor
    if same_scale:
        factors[...] = factors.min()

    # resize images according to calculated scale factor
    cell_size_ws = [cell_size[0]] * n_cols
    cell_size_hs = [cell_size[1]] * n_rows
    for i, img in enumerate(image_list):
        fac = factors[i]
        nsz = [int(round(fac * x)) for x in img.size]
        img = img.resize(nsz)
        image_list[i] = img
        if cell_size[0] <= 0:
            cell_size_ws[i % n_cols] = max(cell_size_ws[i % n_cols], nsz[0])
        if cell_size[1] <= 0:
            cell_size_hs[i // n_cols] = max(cell_size_hs[i // n_cols], nsz[1])

    w = sum(cell_size_ws) + pad * n_cols
    h = sum(cell_size_hs) + pad * n_rows
    background = Image.new("RGB", (w, h), back_color)
    cell_size_ws_sum = np.cumsum([0] + cell_size_ws)
    cell_size_hs_sum = np.cumsum([0] + cell_size_hs)

    # paint images
    for i, img in enumerate(image_list):
        r, c = divmod(i, n_cols)
        x = cell_size_ws_sum[c] + c * pad + pad // 2
        y = cell_size_hs_sum[r] + r * pad + pad // 2
        if align == 0:
            x += (cell_size_ws[c] - img.size[0]) // 2
            y += (cell_size_hs[r] - img.size[1]) // 2
        elif align == 1:
            pass
        elif align == 2:
            x += cell_size_ws[c] - img.size[0]
            y += cell_size_hs[r] - img.size[1]
        else:
            raise ValueError("Unknown align mode.")
        background.paste(img, (x, y))

    return background
