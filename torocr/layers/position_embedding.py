import math
import torch
import torch.nn as nn


class SinCosPositionEmbedding(nn.Module):
    def __init__(self, n_dim, max_len=512):
        super(SinCosPositionEmbedding, self).__init__()

        self.max_len = max_len

        pe = torch.zeros(max_len, n_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_dim, 2).float() * (-math.log(10000.0) / n_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, batch_size, seq_len):
        return self.pe[:, :seq_len, :].repeat((batch_size, 1, 1))
