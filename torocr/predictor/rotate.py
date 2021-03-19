import cv2
import sys
import math
import copy
import torch
import numpy as np

from addict import Dict

from torocr.postprocess.db_postprocess import DBPostProcess
from torocr.algorithm.text_detection import TextDetectionModel
from torocr.algorithm.rowtext_orientation import RowTextOrientationModel


class RotatePredictor(object):
    def __init__(self, config=None):
        self.cfg = Dict(config)

        self.det_model_config = self.cfg.get(
            "det_model_config",
            {
                "backbone": {
                    "type": "MobileNetV3",
                    "args": {
                    }
                },
                "head": {
                    "type": "DBHead",
                    "args": {
                    }
                }
            }
        )

        self.rto_model_config = self.cfg.get(
            "rto_model_config",
            {
                "backbone": {
                    "type": "MobileNetV3",
                    "args": {
                        "in_channels": 3, "scale": 0.35, "model_name": "small"
                    }
                },
                "head": {
                    "type": "CLSHead",
                    "args": {
                        "in_channels": 200, "n_class": 2
                    }
                }
            }
        )
        self.det_resume = self.cfg.get("det_resume", False)
        self.rto_resume = self.cfg.get("rto_resume", False)

        if torch.cuda.is_available():
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.rto_model = RowTextOrientationModel(self.rto_model_config)
        self.det_model = TextDetectionModel(self.det_model_config)

        if self.det_resume:
            self.det_model.load_state_dict(torch.load(self.det_resume, map_location=self.device))
            print(f"load model from {self.det_resume} success !")

        if self.rto_resume:
            self.rto_model.load_state_dict(torch.load(self.rto_resume, map_location=self.device))
            print(f"load model from {self.rto_resume} success !")

        self.det_model = self.det_model.to(self.device)
        self.det_model.eval()

        self.rto_model = self.rto_model.to(self.device)
        self.rto_model.eval()

        self.postprocess = DBPostProcess(
            dict(
                thresh=0.3, box_thresh=0.1,
                max_candidates=1000, unclip_ratio=1.5
            )
        )

        self.cluster_image_w = [64, 128, 192, 256, 320, 512, 768, 1024, 1536, "xxlarge"]
        self.max_batch_size = 128
        self.max_side_len = 1280
        self.label = ["0", "180"]

    def resize_normalize(self, img, max_wh_ratio):
        imgC, imgH, imgW = 3, 48, 192

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

    def rto_call(self, norm_img_batch):
        norm_img_batch = norm_img_batch.copy()
        norm_img_batch = torch.from_numpy(norm_img_batch)
        norm_img_batch = norm_img_batch.to(self.device)

        batch_preds = self.rto_model(norm_img_batch)
        batch_preds = batch_preds.softmax(dim=1).detach().cpu().numpy()
        output = []
        for pred in batch_preds:
            output.append((self.label[int(np.argmax(pred))], np.max(pred)))

        return output

    def resize_normalize(self, img, max_wh_ratio):
        imgC, imgH, imgW = 3, 48, 192

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

    def get_rotate_crop_image(self, img, points):
        points = np.array(points, np.float32)
        left = int(math.floor(np.min(points[:, 0])))
        right = int(math.ceil(np.max(points[:, 0])))
        top = int(math.floor(np.min(points[:, 1])))
        bottom = int(math.ceil(np.max(points[:, 1])))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        img_crop_width = int(np.linalg.norm(points[0] - points[1]))
        img_crop_height = int(np.linalg.norm(points[0] - points[3]))
        pts_std = np.float32([
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height]
        ])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img_crop,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE
        )
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        is_rot_90 = False
        if dst_img_height * 1.0 / dst_img_width >= 1:
            dst_img = np.rot90(dst_img)
            is_rot_90 = True
        return dst_img, is_rot_90

    def det_resize(self, im):
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
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return im, (ratio_h, ratio_w)

    @staticmethod
    def det_normalize(im):
        img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        img_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        im = im.astype(np.float32, copy=False)
        im = im / 255
        im -= img_mean
        im /= img_std
        im = im.transpose((2, 0, 1))
        return im

    def order_points_clockwise(self, pts):
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def predict(self, image):
        ori_im = image.copy()
        im, ratio_list = self.det_resize(image)
        im = self.det_normalize(im)
        im = np.expand_dims(im, 0)
        im = torch.from_numpy(im)
        im = im.to(self.device)
        # print(im.dtype)

        with torch.no_grad():
            preds = self.det_model(im)
            output = self.postprocess(preds, [ratio_list])

            text_rbox = self.filter_tag_det_res(output[0], ori_im.shape)

            img_crop_list = []
            is_rot_90_list = []

            for bno in range(len(text_rbox)):
                tmp_box = copy.deepcopy(text_rbox[bno])
                img_crop, is_rot_90 = self.get_rotate_crop_image(ori_im, tmp_box)
                img_crop_list.append(img_crop)
                is_rot_90_list.append(is_rot_90)

            resized_images = self._preprocess(img_crop_list)
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
                    batch_results = self.rto_call(
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
            # print(text_contents)
            if len(text_contents) == 0:
                return {"text_angle": 0}

            a = np.argmax(np.bincount(text_contents))
            if np.sum(is_rot_90_list) > (len(is_rot_90_list) // 2):

                if a == 180:
                    angle = 90
                else:
                    angle = 270
            else:
                if a == 180:
                    angle = 180
                else:
                    angle = 0
        return {"text_angle": angle}
