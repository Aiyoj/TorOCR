from torch import nn
from addict import Dict

from torocr.layers.activation import HSwish


class ConvBNACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, act=None):
        super(ConvBNACT, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        if act == "relu":
            self.act = nn.ReLU()
        elif act == "hard_swish":
            self.act = HSwish()
        elif act is None:
            self.act = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class ConvBNACTWithPool(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1, act=None):
        super(ConvBNACTWithPool, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=1, padding=(kernel_size - 1) // 2, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        if act is None:
            self.act = None
        else:
            self.act = nn.ReLU()

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class ShortCut(nn.Module):
    def __init__(self, in_channels, out_channels, stride, name, if_first=False):
        super(ShortCut, self).__init__()
        assert name is not None, "shortcut must have name"

        self.name = name
        if in_channels != out_channels or stride != 1:
            if if_first:
                self.conv = ConvBNACT(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                    stride=stride, padding=0, groups=1, act=None
                )
            else:
                self.conv = ConvBNACTWithPool(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=1, groups=1, act=None
                )
        elif if_first:
            self.conv = ConvBNACT(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                stride=stride, padding=0, groups=1, act=None
            )
        else:
            self.conv = None

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        return x


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, if_first, name):
        super(BottleneckBlock, self).__init__()
        assert name is not None, "bottleneck must have name"
        self.name = name
        self.conv0 = ConvBNACT(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, groups=1, act="relu"
        )
        self.conv1 = ConvBNACT(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3,
            stride=stride, padding=1, groups=1, act="relu"
        )
        self.conv2 = ConvBNACT(
            in_channels=out_channels, out_channels=out_channels * 4, kernel_size=1,
            stride=1, padding=0, groups=1, act=None
        )
        self.shortcut = ShortCut(
            in_channels=in_channels, out_channels=out_channels * 4, stride=stride,
            if_first=if_first, name=f"{name}_branch1"
        )
        self.relu = nn.ReLU()
        self.output_channels = out_channels * 4

    def forward(self, x):
        y = self.conv0(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = y + self.shortcut(x)
        return self.relu(y)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, if_first, name):
        super(BasicBlock, self).__init__()
        assert name is not None, "block must have name"
        self.name = name

        self.conv0 = ConvBNACT(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3,
            stride=stride, padding=1, groups=1, act="relu"
        )
        self.conv1 = ConvBNACT(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, groups=1, act=None
        )
        self.shortcut = ShortCut(
            in_channels=in_channels, out_channels=out_channels, stride=stride, name=f"{name}_branch1", if_first=if_first
        )
        self.relu = nn.ReLU()
        self.output_channels = out_channels

    def forward(self, x):
        y = self.conv0(x)
        y = self.conv1(y)
        y = y + self.shortcut(x)
        return self.relu(y)


class ResNetVd(nn.Module):
    def __init__(self, config=None):
        """
        the Resnet backbone network for detection module.
        Args:
            params(dict): the super parameters for network build
        """
        super(ResNetVd, self).__init__()

        self.cfg = Dict(config)

        self.in_channels = self.cfg.get("in_channels", 3)
        self.layers = self.cfg.get("layers", 18)

        supported_layers = {
            18: {"depth": [2, 2, 2, 2], 'block_class': BasicBlock},
            34: {"depth": [3, 4, 6, 3], 'block_class': BasicBlock},
            50: {"depth": [3, 4, 6, 3], 'block_class': BottleneckBlock},
            101: {"depth": [3, 4, 23, 3], 'block_class': BottleneckBlock},
            152: {"depth": [3, 8, 36, 3], 'block_class': BottleneckBlock},
            200: {"depth": [3, 12, 48, 3], 'block_class': BottleneckBlock}
        }
        assert self.layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, self.layers)
        depth = supported_layers[self.layers]["depth"]
        block_class = supported_layers[self.layers]["block_class"]

        num_filters = [64, 128, 256, 512]
        self.conv1 = nn.Sequential(
            ConvBNACT(in_channels=self.in_channels, out_channels=32, kernel_size=3, stride=2, padding=1, act="relu"),
            ConvBNACT(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, act="relu"),
            ConvBNACT(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, act="relu")
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stages = nn.ModuleList()
        self.out_channels = []
        in_ch = 64
        for block_index in range(len(depth)):
            block_list = []
            for i in range(depth[block_index]):
                if self.layers >= 50:
                    if self.layers in [101, 152, 200] and block_index == 2:
                        if i == 0:
                            conv_name = "res" + str(block_index + 2) + "a"
                        else:
                            conv_name = "res" + str(block_index + 2) + "b" + str(i)
                    else:
                        conv_name = "res" + str(block_index + 2) + chr(97 + i)
                else:
                    conv_name = f"res{str(block_index + 2)}{chr(97 + i)}"
                block_list.append(
                    block_class(
                        in_channels=in_ch, out_channels=num_filters[block_index],
                        stride=2 if i == 0 and block_index != 0 else 1,
                        if_first=block_index == i == 0, name=conv_name
                    )
                )
                in_ch = block_list[-1].output_channels
            self.out_channels.append(in_ch)
            self.stages.append(nn.Sequential(*block_list))

    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.pool1(x)

        for stage in self.stages:
            x = stage(x)
            out.append(x)
        return out
