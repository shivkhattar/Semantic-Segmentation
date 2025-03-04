import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.act import build_activation_layer
from utils.normalization import build_normalization_layer
from models.init_weights import init_weights

logger = logging.getLogger()


class PPM(nn.Module):
    def __init__(self, in_channels, out_channels, bins, from_layer, to_layer,
                 mode='bilinear', align_corners=True, norm_cfg=None, act_cfg=None):
        super(PPM, self).__init__()
        self.from_layer = from_layer
        self.to_layer = to_layer
        self.mode = mode
        self.align_corners = align_corners

        if norm_cfg is None:
            norm_cfg = dict(type='BN')

        if act_cfg is None:
            act_cfg = dict(type='Relu', inplace=True)

        self.blocks = nn.ModuleList()
        for bin_ in bins:
            self.blocks.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(bin_),
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    build_normalization_layer(norm_cfg, out_channels, layer_only=True),
                    build_activation_layer(act_cfg, out_channels, layer_only=True)
                )
            )
        logger.info('initializing PPM weights')
        init_weights(self.modules())

    def forward(self, feats):
        feats_ = feats.copy()
        x = feats_[self.from_layer]
        h, w = x.shape[2:]
        out = [x]
        for block in self.blocks:
            feat = F.interpolate(block(x), (h, w), mode=self.mode, align_corners=self.align_corners)
            out.append(feat)
        out = torch.cat(out, 1)
        feats_[self.to_layer] = out
        return feats_
