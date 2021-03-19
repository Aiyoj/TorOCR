import torch
import torch.nn as nn

from torocr.layers.activation import HSwish


class ConvBNACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 act=None):
        super(ConvBNACT, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9)
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


class IBNorm(nn.Module):
    def __init__(self, in_channels):
        super(IBNorm, self).__init__()
        self.bnorm_channels = int(in_channels / 2)
        self.inorm_channels = in_channels - self.bnorm_channels

        self.bnorm = nn.BatchNorm2d(self.bnorm_channels, affine=True)
        self.inorm = nn.InstanceNorm2d(self.inorm_channels, affine=False)

    def forward(self, x):
        bn_x = self.bnorm(x[:, :self.bnorm_channels, ...].contiguous())
        in_x = self.inorm(x[:, self.bnorm_channels:, ...].contiguous())

        return torch.cat((bn_x, in_x), 1)


class ConvIBNACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 act=None):
        super(ConvIBNACT, self).__init__()

        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "hard_swish":
            self.act = HSwish()
        elif act is None:
            self.act = None

        layers = [
            nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation,
                groups=groups, bias=bias
            )
        ]

        layers.append(IBNorm(out_channels))

        if self.act:
            layers.append(self.act)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
