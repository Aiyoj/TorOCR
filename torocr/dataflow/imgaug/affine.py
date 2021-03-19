import os
import cv2
import math
import numpy as np

from datetime import datetime


class Affine(object):
    def __init__(self, rotate=(-20, 20), scale=None, translate_frac=None, shear=0.0, seed=None):
        self.scale = scale
        self.rotate = rotate
        self.shear = shear
        self.translate_frac = translate_frac

        if seed is None:
            seed = (id(self) + os.getpid() + int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295
        self.rng = np.random.RandomState(seed)

    def _rand_range(self, low=1.0, high=None, size=None):
        """
        Generate uniform float random number between low and high using `self.rng`.
        """
        if high is None:
            low, high = 0, low
        if size is None:
            size = []
        return self.rng.uniform(low, high, size).astype("float32")

    def __call__(self, data):
        """
        first Translation, then rotate, final scale.
                [sx, 0, 0]       [cos(theta), -sin(theta), 0]       [1, 0, dx]       [x]
                [0, sy, 0] (dot) [sin(theta),  cos(theta), 0] (dot) [0, 1, dy] (dot) [y]
                [0,  0, 1]       [         0,           0, 1]       [0, 0,  1]       [1]
        :param data:
        :return:
        """

        assert "image" in data and "label" in data, "`image` and `label` in data is required by this process"

        image = data["image"]
        label = data["label"]

        height, width = image.shape[:2]

        if self.scale is not None:
            scale = self._rand_range(self.scale[0], self.scale[1])
        else:
            scale = 1.0

        if self.translate_frac is not None:
            max_dx = self.translate_frac[0] * width
            max_dy = self.translate_frac[1] * height
            dx = np.round(self._rand_range(-max_dx, max_dx))
            dy = np.round(self._rand_range(-max_dy, max_dy))
        else:
            dx = 0
            dy = 0

        if self.shear > 0.0:
            shear = self._rand_range(-self.shear, self.shear)
            sin_shear = math.sin(math.radians(shear))
            cos_shear = math.cos(math.radians(shear))
        else:
            sin_shear = 0.0
            cos_shear = 1.0

        center = (np.array([width, height]) * np.array((0.5, 0.5))) - 0.5
        deg = self._rand_range(self.rotate[0], self.rotate[1])

        transform_matrix = cv2.getRotationMatrix2D(tuple(center), float(deg), scale)

        # Apply shear :
        if self.shear > 0.0:
            m00 = transform_matrix[0, 0]
            m01 = transform_matrix[0, 1]
            m10 = transform_matrix[1, 0]
            m11 = transform_matrix[1, 1]
            transform_matrix[0, 1] = m01 * cos_shear + m00 * sin_shear
            transform_matrix[1, 1] = m11 * cos_shear + m10 * sin_shear
            # Add correction term to keep the center unchanged
            tx = center[0] * (1.0 - m00) - center[1] * transform_matrix[0, 1]
            ty = -center[0] * m10 + center[1] * (1.0 - transform_matrix[1, 1])
            transform_matrix[0, 2] = tx
            transform_matrix[1, 2] = ty

        # Apply shift :
        transform_matrix[0, 2] += dx
        transform_matrix[1, 2] += dy

        new_image = cv2.warpAffine(
            image, transform_matrix, (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        new_label = cv2.warpAffine(
            label, transform_matrix, (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        data["image"] = new_image
        data["label"] = new_label

        return data
