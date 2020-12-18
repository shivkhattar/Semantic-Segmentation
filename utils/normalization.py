import torch.nn as nn


def build_normalization_layer(cfg, num_features, layer_only=False):
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()
    layer_type = cfg_.pop('type')
    if layer_type != 'BN':
        raise KeyError('Unrecognized normalization type {}'.format(layer_type))
    name = 'bn'
    requires_grad = cfg_.pop('requires_grad', True)
    layer = nn.BatchNorm2d(num_features)
    for param in layer.parameters():
        param.requires_grad = requires_grad
    if layer_only:
        return layer
    return name, layer
