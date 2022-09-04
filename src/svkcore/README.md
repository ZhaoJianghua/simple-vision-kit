
## svkcore

Core operations for data prepare, result post-process and visualization.

### common

Common operation like data collection, io, numpy operations

```python
from svkcore import common

# collect images/annotations
img_ps = common.find_file_recursive(directory="root", suffixes=[".jpg"], ignore_case=True)
print(img_ps[:1])

# get basename head
hd = common.bsn_head("root/xxx/xxxx.jpg")
print(hd)

# collect examples
examples = common.collect_examples(directory="dir", suffixes_list=[[".jpg"], [".xml"]])
print(examples[:1])
# or use collect_pascal_data to collect pascal format data
examples = common.collect_pascal_data(directory="dir")
print(examples[:1])

# io operations
data = common.load_json("path")
common.save_json(data, "path")
data = common.load_pickle("path")
common.save_pickle(data, "path")
img = common.cv2imread("path")
common.cv2imwrite("path", img)

# numpy operations
indexes = common.ndarray_index(shape=[7, 8])
distances = common.points_distance(points0=[[0, 0]], points1=[[1, 1]])
grids = common.generate_grid(panel_size=[1024, 1024], grid_size=(256, 256),
                             overlap_size=(24, 24))
# common.seg2point(seg, max_diameter: int, min_distance, fb_threshold: float = 0.5,
#                  min_fore_count: int = 1, max_fore_count: int = -1,
#                  avg_fore_score: float = 0.55, distance_weights=(1., 1.))
# common.seg2line(seg, fb_threshold=0.5, smooth_width=3, partition_width=20,
#                 partition_height=30)

```

### shapes

Base shapes which usually be used in image tasks.

```python
from svkcore import shapes

# shapes and operations
point0 = shapes.Point([0, 0])
point1 = shapes.Point([1, 1])
points = shapes.Points([point0, point1])
bndbox0 = points.bounding_box()
bndbox1 = shapes.Box([0, 0, 1, 1])
bsize = bndbox1.bsize()
center = bndbox1.center()
polygon = bndbox1.to_polygon()
mask = bndbox1.to_mask()
bndbox2 = polygon.bounding_box()
area = polygon.area()
```

### annotation

Pascal and coco annotation load, save and convert.

```python
from svkcore import annotation

# load pascal annotation
ann = annotation.DTAnnotation.load("pascal_annotation_file")
for obj in ann:
    print(obj.name)
    print(obj.bndbox)
    print(obj.difficult)
    obj.name = obj.name + "-new"
ann.dump("pascal_annotation_file_new")

# convert pascal data to coco format
dataset = annotation.DTDataset.load_pascal(annotation_paths=[],
                                           image_paths=[])
dataset.dump_coco("coco_format_dataset_file")

```

### visualize

Visualize part for visualize common shapes.
```python
from PIL import Image
from svkcore import visualize

pil_image = Image.new("RGB", [600, 600])
boxes = [[100, 200, 400, 300]]
visualize.draw_boxes(pil_image, boxes=boxes)
visualize.draw_texts(pil_image, xys=[(100, 200)], texts=["box"])
visualize.draw_points(pil_image, points=[[50, 50]])
visualize.draw_lines(pil_image, lines=[[0, 0], [300, 300]])

from svkcore import annotation

ann = annotation.DTAnnotation.load("path")
visualize.draw_annotation(pil_image, ann, name2cls={}, 
                          add_unknown_name=True)

```