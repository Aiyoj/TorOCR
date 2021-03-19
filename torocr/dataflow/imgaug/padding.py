import cv2


class Padding(object):
    def __init__(self):
        pass

    def __call__(self, data):
        assert "image" in data, "`image` in data is required by this process"
        image = data["image"]
        h, w = image.shape[:2]
        max_side_len = max(h, w)

        padding_image = cv2.copyMakeBorder(
            image, 0, max_side_len - h, 0, max_side_len - w, cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )

        data["image"] = padding_image

        return data
