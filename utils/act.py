# modified from mmcv and mmdetection

import torch.nn as nn


def build_activation_layer(cfg, num_features, postfix='', layer_only=False):
    config = cfg.copy()
    layer_type = config.pop('type')
    if layer_type != 'Relu':
        raise KeyError('Unrecognized activate type {}'.format(layer_type))

    assert isinstance(postfix, (int, str))
    name = 'relu' + str(postfix)

    requires_grad = config.pop('requires_grad', True)
    layer = nn.ReLU(**config)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    if layer_only:
        return layer
    else:
        return name, layer
