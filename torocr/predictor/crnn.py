import cv2
import math
import torch
import torch.nn as nn
import numpy as np

from addict import Dict

from torocr.utils.rec_utils import CTCLabelConverter
from torocr.algorithm.text_recognition import TextRecognitionModel


class CRNNPredictor(object):
    def __init__(self, config=None):
        self.cfg = Dict(config)

        self.character_path = self.cfg.get("character_path", "assets/keys_v1.txt")

        self.model_config = self.cfg.get(
            "model_config",
            {
                "backbone": {
                    "type": "MobileNetV3",
                    "args": {
                        "in_channels": 3, "scale": 0.5, "model_name": "small"
                    }
                },
                "head": {
                    "type": "RNNHead",
                    "args": {
                        "in_channels": 288, "hidden_size": 48, "layers": 2, "n_class": 6625
                    }
                }
            }
        )
        self.resume = self.cfg.get("resume", False)

        c = []
        with open(self.character_path, "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode("utf-8").strip("\n").strip("\r\n")
                c += line
        c += [" "]

        self.converter = CTCLabelConverter(c)

        if torch.cuda.is_available():
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = TextRecognitionModel(self.model_config)
        if self.resume:
            self.model.load_state_dict(torch.load(self.resume, map_location=self.device))
            print(f"load model from {self.resume} success !")

        self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.cluster_image_w = [64, 128, 192, 256, 320, 512, 768, 1024, 1536, "xxlarge"]
        self.max_batch_size = 128

    def resize_normalize(self, img, max_wh_ratio):
        imgC, imgH, imgW = 3, 32, 320

        imgW = int(32 * max_wh_ratio)
        h = img.shape[0]
        w = img.shape[1]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))

        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5

        return resized_image

    @staticmethod
    def _pad_img(img, dst_len, h, w):
        pad_len = dst_len - w
        pad_value = 0
        horizontal = np.full((3, h, pad_len), pad_value, dtype=np.uint8)
        img_concat = np.concatenate([img, horizontal], axis=-1)

        return img_concat

    def _preprocess(self, images):
        resized_images = []
        for index, image in enumerate(images):
            h, w = image.shape[:2]
            wh_ratio = w * 1.0 / h
            resized_image = self.resize_normalize(image, wh_ratio)

            padding = True

            if padding:
                _, h1, w1 = resized_image.shape[:3]
                if w1 <= 64:
                    img_concat_ = self._pad_img(resized_image, 64, h1, w1)
                elif w1 <= 128:
                    img_concat_ = self._pad_img(resized_image, 128, h1, w1)
                elif w1 <= 192:
                    img_concat_ = self._pad_img(resized_image, 192, h1, w1)
                elif w1 <= 256:
                    img_concat_ = self._pad_img(resized_image, 256, h1, w1)
                elif w1 <= 320:
                    img_concat_ = self._pad_img(resized_image, 320, h1, w1)
                elif w1 <= 512:
                    img_concat_ = self._pad_img(resized_image, 512, h1, w1)
                elif w1 <= 768:
                    img_concat_ = self._pad_img(resized_image, 768, h1, w1)
                elif w1 <= 1024:
                    img_concat_ = self._pad_img(resized_image, 1024, h1, w1)
                elif w1 <= 1536:
                    img_concat_ = self._pad_img(resized_image, 1536, h1, w1)
                else:
                    img_concat_ = resized_image

                resized_image = img_concat_
            resized_images.append(resized_image)

        return resized_images

    def call(self, norm_img_batch):
        norm_img_batch = norm_img_batch.copy()
        norm_img_batch = torch.from_numpy(norm_img_batch)

        batch_preds = self.model(norm_img_batch)
        batch_preds = batch_preds.softmax(dim=2).detach().cpu().numpy()
        output = self.converter.decode(batch_preds)

        return output

    def resize_normalize(self, img, max_wh_ratio):
        imgC, imgH, imgW = 3, 32, 320

        imgW = int(32 * max_wh_ratio)
        h = img.shape[0]
        w = img.shape[1]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))

        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def predict(self, images):
        assert isinstance(images, list)

        resized_images = self._preprocess(images)
        indices = []
        text_contents = []
        text_probs = []

        batch_images = {key: [] for key in self.cluster_image_w}
        batch_image_indices = {key: [] for key in self.cluster_image_w}

        for idx, resized_image in enumerate(resized_images):
            rim_c, rim_h, rim_w = resized_image.shape
            if rim_w in self.cluster_image_w:
                batch_images[rim_w].append(resized_image)
                batch_image_indices[rim_w].append(idx)
            else:
                batch_images["xxlarge"].append(resized_image)
                batch_image_indices["xxlarge"].append(idx)

        with torch.no_grad():
            for batch_index in batch_images.keys():
                one_batch = batch_images[batch_index]
                one_batch_indices = batch_image_indices[batch_index]
                n_samples = len(one_batch)
                if len(one_batch) == 0:
                    continue
                start = 0
                if batch_index == "xxlarge":
                    max_batch_size = 1
                else:
                    max_batch_size = self.max_batch_size
                if n_samples > max_batch_size:
                    end = max_batch_size
                else:
                    end = n_samples

                while start < n_samples:
                    batch_indices = one_batch_indices[start:end]
                    indices.extend(batch_indices)
                    batch_results = self.call(
                        np.array(one_batch[start:end])
                    )

                    for result in batch_results:
                        text_contents.append(result[0])
                        text_probs.append(result[1])

                    start = end
                    if end + max_batch_size > n_samples:
                        end = n_samples
                    else:
                        end = end + max_batch_size

        indices = np.argsort(np.array(indices))
        text_contents = np.array(text_contents, dtype=np.object)[indices].tolist()
        text_probs = np.array(text_probs, dtype=np.object)[indices].tolist()

        outputs = {
            "text_content": text_contents,
            "text_prob": text_probs
        }

        return outputs

    def predict_v1(self, batch_image):

        img_num = len(batch_image)
        content_res = []
        score_res = []
        batch_num = 30

        with torch.no_grad():
            for beg_img_no in range(0, img_num, batch_num):
                end_img_no = min(img_num, beg_img_no + batch_num)
                norm_img_batch = []
                max_wh_ratio = 0
                for ino in range(beg_img_no, end_img_no):
                    h, w = batch_image[ino].shape[0:2]
                    wh_ratio = w * 1.0 / h
                    max_wh_ratio = max(max_wh_ratio, wh_ratio)
                for ino in range(beg_img_no, end_img_no):
                    norm_img = self.resize_normalize(batch_image[ino], max_wh_ratio)
                    norm_img = norm_img[np.newaxis, :]
                    norm_img_batch.append(norm_img)
                norm_img_batch = np.concatenate(norm_img_batch)
                norm_img_batch = norm_img_batch.copy()
                norm_img_batch = torch.from_numpy(norm_img_batch)

                batch_preds = self.model(norm_img_batch)
                batch_preds = batch_preds.softmax(dim=2).detach().cpu().numpy()
                output = self.converter.decode(batch_preds)

                for c_s in output:
                    content_res.append(c_s[0])
                    score_res.append(c_s[1])

        return {"text_content": content_res, "text_score": score_res}
