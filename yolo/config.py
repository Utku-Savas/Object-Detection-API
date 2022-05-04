from functools import lru_cache

from pydantic import BaseSettings


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


@lru_cache
def get_config():
    return Config()
