import torch.nn as nn


def constant_init(module, val, bias=0):
    nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module, a=0, mode='fan_out', nonlinearity='relu', bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(module.weight,
                                 a=a,
                                 mode=mode,
                                 nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(module.weight,
                                a=a,
                                mode=mode,
                                nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


# adapted from mmdetection and mmcv
def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            kaiming_init(m)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            constant_init(m, 1)
