import torch
import torch.nn as nn

from addict import Dict
from collections import OrderedDict


class DBHead(nn.Module):
    def __init__(self, config=None):
        super(DBHead, self).__init__()

        self.cfg = Dict(config)

        self.k = self.cfg.get("k", 50)
        self.bias = self.cfg.get("bias", False)
        self.in_channels = self.cfg.get("in_channels", [16, 24, 56, 480])
        self.inner_channels = self.cfg.get("inner_channels", 96)
        self.num_classes = self.cfg.get("num_classes", 1)

        self.up5 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up4 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")

        self.in5 = nn.Conv2d(self.in_channels[-1], self.inner_channels, 1, bias=self.bias)
        self.in4 = nn.Conv2d(self.in_channels[-2], self.inner_channels, 1, bias=self.bias)
        self.in3 = nn.Conv2d(self.in_channels[-3], self.inner_channels, 1, bias=self.bias)
        self.in2 = nn.Conv2d(self.in_channels[-4], self.inner_channels, 1, bias=self.bias)

        self.out5 = nn.Sequential(
            nn.Conv2d(self.inner_channels, self.inner_channels // 4, 3, padding=1, bias=self.bias),
            nn.Upsample(scale_factor=8, mode="nearest")
        )
        self.out4 = nn.Sequential(
            nn.Conv2d(self.inner_channels, self.inner_channels // 4, 3, padding=1, bias=self.bias),
            nn.Upsample(scale_factor=4, mode="nearest")
        )
        self.out3 = nn.Sequential(
            nn.Conv2d(self.inner_channels, self.inner_channels // 4, 3, padding=1, bias=self.bias),
            nn.Upsample(scale_factor=2, mode="nearest")
        )
        self.out2 = nn.Conv2d(self.inner_channels, self.inner_channels // 4, 3, padding=1, bias=self.bias)

        self.binarize = nn.Sequential(
            nn.Conv2d(self.inner_channels, self.inner_channels // 4, 3, padding=1, bias=self.bias),
            nn.BatchNorm2d(self.inner_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.inner_channels // 4, self.inner_channels // 4, 2, 2),
            nn.BatchNorm2d(self.inner_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.inner_channels // 4, self.num_classes, 2, 2),
            nn.Sigmoid()
        )

        self.binarize.apply(self.weights_init)

        self.thresh = nn.Sequential(
            nn.Conv2d(self.inner_channels, self.inner_channels // 4, 3, padding=1, bias=self.bias),
            nn.BatchNorm2d(self.inner_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.inner_channels // 4, self.inner_channels // 4, 2, 2),
            nn.BatchNorm2d(self.inner_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.inner_channels // 4, self.num_classes, 2, 2),
            nn.Sigmoid()
        )

        self.thresh.apply(self.weights_init)

        self.in5.apply(self.weights_init)
        self.in4.apply(self.weights_init)
        self.in3.apply(self.weights_init)
        self.in2.apply(self.weights_init)

        self.out5.apply(self.weights_init)
        self.out4.apply(self.weights_init)
        self.out3.apply(self.weights_init)
        self.out2.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def forward(self, features, gt=None, masks=None):
        c2, c3, c4, c5 = features
        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)

        out4 = self.up5(in5) + in4
        out3 = self.up4(out4) + in3
        out2 = self.up3(out3) + in2

        p5 = self.out5(in5)
        p4 = self.out4(out4)
        p3 = self.out3(out3)
        p2 = self.out2(out2)

        fuse = torch.cat((p5, p4, p3, p2), 1)

        binary = self.binarize(fuse)
        if self.num_classes == 1:
            result = OrderedDict(binary=binary)
        else:
            result = OrderedDict()
            for i in range(self.num_classes):
                result.update({"binary_{}".format(i): binary[:, [i], :, :]})

        if self.training:
            thresh = self.thresh(fuse)
            thresh_binary = self.step_function(binary, thresh)
            if self.num_classes == 1:
                result.update(thresh=thresh, thresh_binary=thresh_binary)
            else:
                for i in range(self.num_classes):
                    result.update(
                        {
                            "thresh_{}".format(i): thresh[:, [i], :, :],
                            "thresh_binary_{}".format(i): thresh_binary[:, [i], :, :]
                        }
                    )
        return result

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))
