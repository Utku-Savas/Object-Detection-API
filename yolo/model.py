import os.path

import cv2
import numpy as np
import onnxruntime

from .utils import non_max_suppression, preprocess, read_file, scale_coords


class YOLO:
    def __init__(
        self,
        weight: str,
        is_cuda: bool = False,
        data: str = None,
        imgsz: int = 640,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
    ):

        self.providers = is_cuda
        self.session = weight
        self.imgsz = imgsz
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.classes = data

    def warmup(self):
        im = np.zeros((1, 3, 640, 640), dtype=np.float32)
        onnx_input = {self.__session.get_inputs()[0].name: im}
        onnx_out_name = [self.__session.get_outputs()[0].name]

        _ = self.__session.run(onnx_out_name, onnx_input)[0]

    def __detect(
        self,
        img,
        imgsz=(640, 640),
        conf_thresh=0.25,
        iou_thresh=0.45,
    ):
        img_shape = img.shape

        img = preprocess(img, imgsz)

        onnx_input = {self.__session.get_inputs()[0].name: img}
        onnx_out_name = [self.__session.get_outputs()[0].name]

        outputs = self.__session.run(onnx_out_name, onnx_input)[0]

        outputs = non_max_suppression(outputs, conf_thresh, iou_thresh)
        outputs[0][:, :4] = scale_coords(
            img.shape[2:],
            outputs[0][:, :4],
            img_shape,
        )

        return outputs

    def predict(self, img):
        preds = self.__detect(
            img,
            imgsz=self.__imgsz,
            conf_thresh=self.__conf_thresh,
            iou_thresh=self.__iou_thresh,
        )[0]

        for pred in preds:
            bbox = pred[:4]
            if self.__classes is not None:
                cls = self.__classes[int(pred[5])]
            else:
                cls = int(pred[5])

            r, g, b = (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255),
            )

            xmin, ymin, xmax, ymax = (
                int(bbox[0]),
                int(bbox[1]),
                int(bbox[2]),
                int(bbox[3]),
            )
            cv2.putText(
                img,
                cls,
                (xmin, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (b, g, r),
                2,
            )
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (b, g, r), 2)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _, encoded_img = cv2.imencode(".jpg", img)
        return encoded_img

    @property
    def providers(self):
        return self.__providers

    @providers.setter
    def providers(self, is_cuda: bool):
        boolean_keys = {
            "True": True,
            "true": True,
            "TRUE": True,
            "1": True,
            "Cuda": True,
            "cuda": True,
            "CUDA": True,
            1: True,
            "False": True,
            "false": True,
            "FALSE": True,
            "0": True,
            "Cpu": True,
            "cpu": True,
            "CPU": True,
            0: True,
        }

        if not isinstance(is_cuda, bool):
            if is_cuda not in boolean_keys.keys():
                raise Exception("Wrong keyword: is_cuda")
            else:
                is_cuda = boolean_keys[is_cuda]

        if is_cuda:
            self.__providers = [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
        else:
            self.__providers = ["CPUExecutionProvider"]

    @property
    def session(self):
        return self.__session

    @session.setter
    def session(self, weight: str):
        if not isinstance(weight, str):
            raise Exception("Model path should be a string type: weight")

        elif not os.path.exists(weight):
            raise Exception("Model doesn't exists: weight")

        elif "." not in weight or weight.rsplit(".", 1)[1].lower() != "onnx":
            raise Exception("Use only onnx models")

        else:
            self.__session = onnxruntime.InferenceSession(
                weight, providers=self.__providers
            )

    @property
    def imgsz(self):
        return self.__imgsz

    @imgsz.setter
    def imgsz(self, size: int):
        if not isinstance(size, int):
            if (
                isinstance(size, float)
                or isinstance(size, np.int32)
                or isinstance(size, np.int64)
                or isinstance(size, np.float32)
                or isinstance(size, np.float64)
            ):

                size = int(size)
                self.__imgsz = (size, size)
            else:
                raise Exception("Size must be an integer type: imgsz")

        else:
            self.__imgsz = (size, size)

    @property
    def conf_thresh(self):
        return self.__conf_thresh

    @conf_thresh.setter
    def conf_thresh(self, confidence: float):
        if not isinstance(confidence, float):
            if isinstance(confidence, np.float32) or isinstance(confidence, np.float64):
                confidence = float(confidence)
                self.__conf_thresh = confidence
            else:
                raise Exception(
                    "Confidence threshold must be an float type: conf_thresh"
                )
        else:
            self.__conf_thresh = confidence

    @property
    def iou_thresh(self):
        return self.__iou_thresh

    @iou_thresh.setter
    def iou_thresh(self, iou: float):
        if not isinstance(iou, float):
            if isinstance(iou, np.float32) or isinstance(iou, np.float64):
                iou = float(iou)
                self.__iou_thresh = iou
            else:
                raise Exception(
                    "Confidence threshold must be an float type: iou_thresh"
                )
        else:
            self.__iou_thresh = iou

    @property
    def classes(self):
        return self.__classes

    @classes.setter
    def classes(self, data: str):
        if not isinstance(data, str):
            self.__classes = None

        elif not os.path.exists(data):
            self.__classes = None

        else:
            lines = read_file(data)
            self.__classes = lines
