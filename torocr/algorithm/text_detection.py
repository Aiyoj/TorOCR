import torch.nn as nn

from addict import Dict

from torocr.modeling.head.db_head import DBHead
from torocr.modeling.backbone.det_resnet_vd import ResNetVd
from torocr.modeling.backbone.det_mobilenet_v3 import MobileNetV3


# from torocr.modeling.backbone.det_deform_resnet import DeformResNet


class TextDetectionModel(nn.Module):
    def __init__(self, config=None):
        super(TextDetectionModel, self).__init__()

        self.cfg = Dict(config)

        self.backbone_type = self.cfg.backbone.get("type", "MobileNetV3")
        self.backbone_args = self.cfg.backbone.get(
            "args", {"in_channels": 3, "scale": 0.5, "model_name": "large", "disable_se": True}
        )
        self.head_type = self.cfg.head.get("type", "DBHead")
        self.head_args = self.cfg.head.get(
            "args", {"in_channels": [16, 24, 56, 480], "inner_channels": 96, "k": 50, "bias": False, "num_classes": 1}
        )

        self.backbone = eval(self.backbone_type)(self.backbone_args)
        self.head = eval(self.head_type)(self.head_args)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
