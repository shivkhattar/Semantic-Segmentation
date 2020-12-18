import logging

import torch.nn as nn
from models.init_weights import init_weights
from models.brickbuilder import brick, bricks

logger = logging.getLogger()


class GFPN(nn.Module):
    def __init__(self, neck, fusion=None):
        super().__init__()
        self.neck = bricks(neck)
        self.fusion = brick(fusion) if fusion else None
        logger.info('initializing GFPN weights')
        init_weights(self.modules())

    def forward(self, bottom_up):
        x = None
        feats = {}
        for ii, layer in enumerate(self.neck):
            top_down = layer.start_layer.get('top_down')
            lateral = layer.start_layer.get('lateral')

            if lateral:
                ll = bottom_up[lateral]
            else:
                ll = None
            if top_down is None:
                td = None
            elif 'c' in top_down:
                td = bottom_up[top_down]
            elif 'p' in top_down:
                td = feats[top_down]
            else:
                raise ValueError('Key error')

            x = layer(td, ll)
            feats[layer.to_layer] = x
        if self.fusion:
            x = self.fusion(feats)
        return x
