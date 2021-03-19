import torch
import torch.nn as nn

from addict import Dict
from collections import OrderedDict


class CLSHead(nn.Module):
    def __init__(self, config=None):
        super(CLSHead, self).__init__()

        self.cfg = Dict(config)
        self.n_class = self.cfg.get("n_class", 2)
        self.in_channels = self.cfg.get("in_channels", 200)

        self.fc = nn.Linear(self.in_channels, self.n_class)

        self.gap = nn.AdaptiveMaxPool2d((1, 1))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        pool = self.gap(inputs)
        pool = pool.reshape((-1, self.in_channels))
        out = self.fc(pool)
        out = self.softmax(out)

        return out
