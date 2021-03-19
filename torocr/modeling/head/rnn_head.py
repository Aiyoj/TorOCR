import torch
import torch.nn as nn

from addict import Dict


class Reshape(nn.Module):
    def __init__(self, in_channels):
        super(Reshape, self).__init__()

        self.out_channels = in_channels

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w)
        x = x.permute((0, 2, 1))
        return x


class RNNHead(nn.Module):
    def __init__(self, config=None):
        super(RNNHead, self).__init__()

        self.cfg = Dict(config)

        self.in_channels = self.cfg.get("in_channels", 288)
        self.hidden_size = self.cfg.get("hidden_size", 48)
        self.layers = self.cfg.get("layers", 2)
        self.n_class = self.cfg.get("n_class", 6625)

        self.reshape = Reshape(self.in_channels)

        self.lstm0 = nn.LSTM(
            self.in_channels, self.hidden_size, bidirectional=False, batch_first=True, num_layers=2
        )
        self.lstm1 = nn.LSTM(
            self.in_channels, self.hidden_size, bidirectional=False, batch_first=True, num_layers=2
        )

        self.fc = nn.Linear(self.hidden_size * 2, self.n_class)

    def forward(self, x):
        x = self.reshape(x)
        x1, _ = self.lstm0(x)
        reverse_x = torch.flip(x, [1])
        x2, _ = self.lstm1(reverse_x)
        x2 = torch.flip(x2, [1])
        x = torch.cat([x1, x2], 2)
        x = self.fc(x)

        return x
