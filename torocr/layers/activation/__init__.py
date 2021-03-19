import torch.nn as nn
import torch.nn.functional as F


class HSwish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class HSigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(1.2 * x + 3, inplace=True) / 6
        return out
