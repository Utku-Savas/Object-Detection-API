## Object Detection API

Returns image with detected bounding boxes. YOLOv5[https://github.com/ultralytics/yolov5] is used for object detection. Supports only **ONNX** models.

### Installation

```bash
git clone https://github.com/Utku-Savas/Object-Detection-API.git  # clone
cd Object-Detection-API
```

### Usage

Copy ONNX model file and classes list do data folder.
```bash
cp yolov5s.onnx <project folder>/data/
cp classes.txt <project folder>/data/
```

Change default config parameters

> yolo/config.py
```python
class Config(BaseSettings):
    classes_path: str = "./data/classes.txt"
    conf_threshold: float = 0.5
    cuda: bool = False
    host: str = "0.0.0.0"
    image_size: int = 640
    iou_threshold: float = 0.45
    log_level: str = "info"
    onnx_path: str = "./data/yolov5s.onnx"
    port: int = "5000"
```

Run docker image
```bash
docker-compose up
```

### Sample Mock Code

```python
import requests
import io

import cv2
import numpy as np
from PIL import Image

url = "http://0.0.0.0:5000/"
files = {'file': open('bus.jpg', 'rb')}

r = requests.post(url, files=files)

img = numpy.array(Image.open(io.BytesIO(r.content)))

cv2.imshow("IMG", img)
```
