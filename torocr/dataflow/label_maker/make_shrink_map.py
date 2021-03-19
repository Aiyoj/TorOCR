import cv2
import pyclipper
import numpy as np

from shapely.geometry import Polygon


class MakeShrinkMap(object):

    def __init__(self, min_text_size=8, shrink_ratio=0.4):
        self.min_text_size = min_text_size
        self.shrink_ratio = shrink_ratio

    def __call__(self, data, *args, **kwargs):
        polygons = data["polygons"]
        ignore_tags = data["ignore_tags"]
        image = data["image"]
        filename = data["filename"]

        h, w = image.shape[:2]
        polygons, ignore_tags = self.validate_polygons(polygons, ignore_tags, h, w)
        gt = np.zeros((1, h, w), dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)
        for i in range(polygons.shape[0]):
            polygon = polygons[i]
            height = min(
                np.linalg.norm(polygon[0] - polygon[3]),
                np.linalg.norm(polygon[1] - polygon[2])
            )
            width = min(
                np.linalg.norm(polygon[0] - polygon[1]),
                np.linalg.norm(polygon[2] - polygon[3])
            )
            if ignore_tags[i] or min(height, width) < self.min_text_size:
                cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
                ignore_tags[i] = True
            else:
                polygon_shape = Polygon(polygon)
                distance = polygon_shape.area * (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
                subject = [tuple(l) for l in polygons[i]]
                padding = pyclipper.PyclipperOffset()
                padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                shrinked = padding.Execute(-distance)
                if shrinked == []:
                    cv2.fillPoly(mask, polygon.astype(
                        np.int32)[np.newaxis, :, :], 0)
                    ignore_tags[i] = True
                    continue
                shrinked = np.array(shrinked[0]).reshape(-1, 2)
                cv2.fillPoly(gt[0], [shrinked.astype(np.int32)], 1)

        if filename is None:
            filename = ""
        data.update(image=image, polygons=polygons, gt=gt, mask=mask, filename=filename)
        return data

    def validate_polygons(self, polygons, ignore_tags, h, w):
        # if len(polygons) == 0:
        if polygons.shape[0] == 0:
            return polygons, ignore_tags
        assert polygons.shape[0] == len(ignore_tags)

        polygons[:, :, 0] = np.clip(polygons[:, :, 0], 0, w - 1)
        polygons[:, :, 1] = np.clip(polygons[:, :, 1], 0, h - 1)

        for i in range(polygons.shape[0]):
            area = self.polygon_area(polygons[i])
            if abs(area) < 1:
                ignore_tags[i] = True
            if area > 0:
                polygons[i] = polygons[i][(0, 3, 2, 1), :]
        return polygons, ignore_tags

    def polygon_area(self, polygon):
        edge = [
            (polygon[1][0] - polygon[0][0]) * (polygon[1][1] + polygon[0][1]),
            (polygon[2][0] - polygon[1][0]) * (polygon[2][1] + polygon[1][1]),
            (polygon[3][0] - polygon[2][0]) * (polygon[3][1] + polygon[2][1]),
            (polygon[0][0] - polygon[3][0]) * (polygon[0][1] + polygon[3][1])
        ]
        return np.sum(edge) / 2.
