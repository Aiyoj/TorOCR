import numpy as np


class NormalizeImageV1(object):
    def __init__(self):
        self.img_mean = np.array([122.67891434, 116.66876762, 104.00698793], dtype=np.float32)

    def __call__(self, data):
        assert "image" in data, "`image` in data is required by this process"
        image = data["image"]
        image = image.astype(np.float32, copy=False)
        image -= self.img_mean
        image /= 255.
        image = np.transpose(image, [2, 0, 1])
        data["image"] = image

        return data


class NormalizeImageV2(object):
    def __init__(self):
        self.img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.img_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __call__(self, data):
        assert "image" in data, "`image` in data is required by this process"
        image = data["image"]
        # thresh_map = data["thresh_map"]
        # thresh_mask = data["thresh_mask"]
        # gt = np.squeeze(data["gt"], axis=0)
        # print(gt.shape)
        # mask = data["mask"]
        # flag = random.choice([0, 1, 2, 3])
        # if flag == 0:
        #     pass
        # elif flag == 1:
        #     image = np.rot90(image, k=1)
        #     thresh_map = np.rot90(thresh_map, k=1)
        #     thresh_mask = np.rot90(thresh_mask, k=1)
        #     gt = np.rot90(gt, k=1)
        #     mask = np.rot90(mask, k=1)
        # elif flag == 2:
        #     image = np.rot90(image, k=2)
        #     thresh_map = np.rot90(thresh_map, k=2)
        #     thresh_mask = np.rot90(thresh_mask, k=2)
        #     gt = np.rot90(gt, k=2)
        #     mask = np.rot90(mask, k=2)
        # else:
        #     image = np.rot90(image, k=3)
        #     thresh_map = np.rot90(thresh_map, k=3)
        #     thresh_mask = np.rot90(thresh_mask, k=3)
        #     gt = np.rot90(gt, k=3)
        #     mask = np.rot90(mask, k=3)
        # print(image.shape, data["thresh_map"].shape, data["thresh_mask"].shape, data["gt"].shape, data["mask"].shape)
        image = image.astype(np.float32, copy=False)
        image /= 255
        image -= self.img_mean
        image /= self.img_std
        image = np.transpose(image, [2, 0, 1])
        data["image"] = image

        return data
