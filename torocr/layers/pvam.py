import torch
import torch.nn as nn

from torocr.layers.position_embedding import SinCosPositionEmbedding


class PVAM(nn.Module):
    def __init__(self, n_dim=256, seq_len=26):
        super(PVAM, self).__init__()
        self.n_dim = n_dim
        self.seq_len = seq_len
        self.fo_embedding = SinCosPositionEmbedding(n_dim, 256)
        self.wo = nn.Linear(n_dim, n_dim)
        self.wv = nn.Linear(n_dim, n_dim)
        self.we = nn.Linear(n_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x, seq_len):
        batch_size = x.size(0)
        x = x.permute(0, 2, 3, 1)
        reading_state = self.fo_embedding(batch_size, seq_len)
        attention_r = self.wo(reading_state)
        attention_r = attention_r.unsqueeze(2).unsqueeze(2)
        attention_x = self.wv(x)
        attention_x = attention_x.unsqueeze(1)

        # Score function for Attention Mechanism
        att = self.tanh(attention_x + attention_r)  # [b, seq_len, h*w, c]
        att = self.we(att)

        att = att.reshape((batch_size, seq_len, -1))

        alpha = self.softmax(att)
        alpha = alpha.unsqueeze(3)

        x = attention_x.reshape((batch_size, 1, -1, self.n_dim)) * alpha
        x = torch.mean(x, dim=2)
        return x
