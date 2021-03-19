import cv2
import random
import numpy as np


class AugHSV(object):
    def __init__(self, p=0.5, hgain=0.5, sgain=0.5, vgain=0.5):
        self.p = p
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

        # hsv_h: 0.015  # image HSV-Hue augmentation (fraction)
        # hsv_s: 0.7  # image HSV-Saturation augmentation (fraction)
        # hsv_v: 0.4  # image HSV-Value augmentation (fraction)

        # hsv_h: 0.0138
        # hsv_s: 0.664
        # hsv_v: 0.464

    def __call__(self, data):
        assert "image" in data, "`image` in data is required by this process"

        if random.random() < 1 - self.p:
            return data

        image = data["image"]
        r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1

        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
        dtype = image.dtype  # uint8

        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        data["image"] = img

        return data
