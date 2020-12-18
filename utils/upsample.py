import torch.nn as nn
import torch.nn.functional as F


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, scale_bias=0,
                 mode='nearest', align_corners=None):
        super(Upsample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.scale_bias = scale_bias
        self.mode = mode
        self.align_corners = align_corners
        assert (self.size is None) ^ (self.scale_factor is None)

    def forward(self, x):
        if self.size:
            size = self.size
        else:
            n, c, h, w = x.size()
            new_height = int(h * self.scale_factor + self.scale_bias)
            new_width = int(w * self.scale_factor + self.scale_bias)
            size = (new_height, new_width)
        return F.interpolate(x, size=size, mode=self.mode,
                             align_corners=self.align_corners)

    def extra_repr(self):
        out = 'scale_factor=' + str(self.scale_factor) + ', scale_bias=' + str(
            self.scale_bias) if self.size is None else 'size=' + str(self.size)
        out += ', mode=' + self.mode
        return out
