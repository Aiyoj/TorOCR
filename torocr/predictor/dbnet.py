import sys
import cv2
import torch
import numpy as np

from addict import Dict

from torocr.algorithm.text_detection import TextDetectionModel
from torocr.postprocess.db_postprocess import DBPostProcess


class DBNetPrediction(object):
    def __init__(self, config=None):
        self.cfg = Dict(config)

        self.model_config = self.cfg.get(
            "model_config",
            {
                "backbone": {
                    "type": "DeformResNet",
                    "args": {
                        "layers": 50,
                    }
                },
                "head": {
                    "type": "DBHead",
                    "args": {
                        "k": 50,
                        "bias": False,
                        "inner_channels": 256,
                        "in_channels": [256, 512, 1024, 2048],
                        "num_classes": 5
                    }
                }
            }
        )
        self.polygon = self.cfg.get("polygon", False)
        self.resume = self.cfg.get("resume", False)
        self.max_side_len = self.cfg.get("max_side_len", 2400)
        self.mag_ratio = self.cfg.get("mag_ratio", 1.5)
        self.thresh = self.cfg.get("thresh", 0.3)
        self.box_thresh = self.cfg.get("box_thresh", 0.1)
        self.max_candidates = self.cfg.get("max_candidates", 1000)
        self.unclip_ratio = self.cfg.get("unclip_ratio", 2.0)
        self.polygon = self.cfg.get("polygon", False)

        if torch.cuda.is_available():
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = TextDetectionModel(self.model_config)
        if self.resume:
            self.model.load_state_dict(torch.load(self.resume, map_location=torch.device("cpu")))
            print(f"load model from {self.resume} success !")

        self.model = self.model.to(self.device)
        self.model.eval()

        self.postprocess = DBPostProcess(
            dict(
                thresh=self.thresh, box_thresh=self.box_thresh, polygon=self.polygon,
                max_candidates=self.max_candidates, unclip_ratio=self.unclip_ratio
            )
        )

    @staticmethod
    def normalize(im):
        img_mean = np.array([122.67891434, 116.66876762, 104.00698793], np.float32)
        im = im.astype(np.float32, copy=False)
        im -= img_mean
        im = im / 255
        im = im.transpose((2, 0, 1))
        return im

    @staticmethod
    def normalize_v2(im):
        img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        img_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        im = im.astype(np.float32, copy=False)
        im = im / 255
        im -= img_mean
        im /= img_std
        im = im.transpose((2, 0, 1))
        return im

    def resize_v2(self, im):
        max_side_len = self.max_side_len
        height, width, channel = im.shape

        # Magnify image size
        target_size = max(height, width) * self.mag_ratio

        # Set original image size
        if target_size > max_side_len:
            target_size = max_side_len

        ratio = target_size / max(height, width)

        target_h, target_w = int(height * ratio), int(width * ratio)

        proc = cv2.resize(im, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Make canvas and paste image
        target_h32, target_w32 = target_h, target_w
        if target_h % 32 != 0:
            target_h32 = target_h + (32 - target_h % 32)
        if target_w % 32 != 0:
            target_w32 = target_w + (32 - target_w % 32)
        resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
        resized[0:target_h, 0:target_w, :] = proc

        return resized, (height, width), (target_h32 - target_h, target_w32 - target_w)

    def resize(self, im):
        """
        resize image to a size multiple of 32 which is required by the network
        args:
            img(array): array with shape [h, w, c]
        return(tuple):
            img, (ratio_h, ratio_w)
        """
        max_side_len = self.max_side_len
        h, w, _ = im.shape

        resize_w = w
        resize_h = h

        # limit the max side
        if max(resize_h, resize_w) > max_side_len:
            if resize_h > resize_w:
                ratio = float(max_side_len) / resize_h
            else:
                ratio = float(max_side_len) / resize_w
        else:
            ratio = 1.
        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)
        if resize_h % 32 == 0:
            resize_h = resize_h
        elif resize_h // 32 <= 1:
            resize_h = 32
        else:
            resize_h = (resize_h // 32 - 1) * 32
        if resize_w % 32 == 0:
            resize_w = resize_w
        elif resize_w // 32 <= 1:
            resize_w = 32
        else:
            resize_w = (resize_w // 32 - 1) * 32
        try:
            if int(resize_w) <= 0 or int(resize_h) <= 0:
                return None, (None, None)
            im = cv2.resize(im, (int(resize_w), int(resize_h)))
        except:
            print(im.shape, resize_w, resize_h)
            sys.exit(0)
        # ratio_h = resize_h / float(h)
        # ratio_w = resize_w / float(w)
        return im, (h, w)

    def predict(self, image):
        im, shape_list, pad_list = self.resize_v2(image)
        im = self.normalize_v2(im)
        im = np.expand_dims(im, 0)
        im = torch.from_numpy(im)
        im = im.to(self.device)

        with torch.no_grad():
            preds = self.model(im)
            if self.model.head.num_classes == 1:
                output = self.postprocess(preds["binary"], [shape_list], [pad_list])
                return {"text_line": output[0]}
            else:
                out = {}
                for i in range(self.model.head.num_classes):
                    output = self.postprocess(preds["binary_{}".format(i)], [shape_list], [pad_list])
                    out.update({"text_line_{}".format(i): output[0]})

                return out
