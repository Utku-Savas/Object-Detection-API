import io

import numpy
from fastapi import Depends, FastAPI, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image

from .config import Config, get_config
from .model import YOLO

app = FastAPI()


@app.post("/")
async def upload(
    file: UploadFile,
    config: Config = Depends(get_config),
):
    ObjectDetectionModel = YOLO(
        weight=config.onnx_path,
        data=config.classes_path,
        is_cuda=config.cuda,
        imgsz=config.image_size,
        conf_thresh=config.conf_threshold,
        iou_thresh=config.conf_threshold,
    )
    ObjectDetectionModel.warmup()
    content = await file.read()
    img = numpy.array(Image.open(io.BytesIO(content)))
    res_data = ObjectDetectionModel.predict(img)
    return StreamingResponse(
        io.BytesIO(res_data),
        media_type="image/jpeg",
        headers={"Content-Disposition": f"inline; filename=\"{file.filename}\""},
    )
