import torch.nn as nn

from addict import Dict

from torocr.modeling.head.cls_head import CLSHead
from torocr.modeling.backbone.rec_mobilenet_v3 import MobileNetV3


class RowTextOrientationModel(nn.Module):
    def __init__(self, config=None):
        super(RowTextOrientationModel, self).__init__()

        self.cfg = Dict(config)

        self.backbone_type = self.cfg.backbone.get("type", "MobileNetV3")
        self.backbone_args = self.cfg.backbone.get(
            "args", {"in_channels": 3, "scale": 0.35, "model_name": "small"}
        )
        self.head_type = self.cfg.head.get("type", "CLSHead")
        self.head_args = self.cfg.head.get(
            "args", {"in_channels": 200, "n_class": 2}
        )

        self.backbone = eval(self.backbone_type)(self.backbone_args)
        self.head = eval(self.head_type)(self.head_args)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
