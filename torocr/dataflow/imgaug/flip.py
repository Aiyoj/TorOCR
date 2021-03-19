import cv2
import random


class MultiFlipLR(object):
    def __init__(self, p=0.5, num_classes=1):
        self.p = p
        self.num_classes = num_classes

    def __call__(self, data):
        assert "image" in data, "`image` in data is required by this process"

        if random.random() < 1 - self.p:
            return data

        image = data["image"]
        label = data["label"]

        h_image = cv2.flip(image, 1)
        h_label = cv2.flip(label, 1)

        data["image"] = h_image
        data["label"] = h_label

        return data
