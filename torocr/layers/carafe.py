import torch
import torch.nn as nn
import torch.nn.functional as F

from torocr.layers.conv_bn_act import ConvBNACT


class CARAFE(nn.Module):
    def __init__(self, c, c_mid=64, scale=2, k_up=5, k_enc=3):
        """ The unofficial implementation of the CARAFE module.
        The details are in "https://arxiv.org/abs/1905.02188".
        Args:
            c: The channel number of the input and the output.
            c_mid: The channel number after compression.
            scale: The expected upsample scale.
            k_up: The size of the reassembly kernel.
            k_enc: The kernel size of the encoder.
        Returns:
            x: The upsampled feature map.
        """
        super(CARAFE, self).__init__()

        self.scale = scale

        self.comp = ConvBNACT(c, c_mid, kernel_size=1, stride=1, padding=0, dilation=1, act="relu")
        self.enc = ConvBNACT(c_mid, (scale * k_up) ** 2, kernel_size=k_enc, stride=1,
                             padding=k_enc // 2, dilation=1, act=None)
        self.pix_shf = nn.PixelShuffle(scale)

        self.upsmp = nn.Upsample(scale_factor=scale, mode="nearest")
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale, padding=k_up // 2 * scale)

    def forward(self, x):
        b, c, h, w = x.size()
        h_, w_ = h * self.scale, w * self.scale

        W = self.comp(x)  # b * m * h * w
        W = self.enc(W)  # b * 100 * h * w
        W = self.pix_shf(W)  # b * 25 * h_ * w_
        W = F.softmax(W, dim=1)  # b * 25 * h_ * w_

        x = self.upsmp(x)  # b * c * h_ * w_
        x = self.unfold(x)  # b * 25c * h_ * w_
        x = x.view(b, c, -1, h_, w_)  # b * 25 * c * h_ * w_

        x = torch.einsum("bkhw,bckhw->bchw", W, x)  # b * c * h_ * w_
        return x
