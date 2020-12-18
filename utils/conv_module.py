import warnings

import torch.nn as nn

from .act import build_activation_layer
from .normalization import build_normalization_layer

conv_cfg = {
    'Conv': nn.Conv2d,
}


def build_conv_layer(cfg, *args, **kwargs):
    config = cfg.copy()
    layer_type = config.pop('type')
    if layer_type not in conv_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        conv_layer = conv_cfg[layer_type]
    return conv_layer(*args, **kwargs, **config)


class ConvModule(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias='auto',
                 conv_cfg=None, norm_cfg=None, act_cfg=None, order=('conv', 'norm', 'act'), dropout=None):
        super(ConvModule, self).__init__()
        if act_cfg is None:
            act_cfg = dict(type='Relu', inplace=True)
        if conv_cfg is None:
            conv_cfg = dict(type='Conv')
        assert isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(['conv', 'norm', 'act'])

        self.with_norm = norm_cfg is not None
        self.with_act = act_cfg is not None
        self.with_dropout = dropout is not None
        if bias == 'auto':
            bias = False if self.with_norm else True
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            warnings.warn('ConvModule has norm and bias at the same time')

        self.conv = build_conv_layer(conv_cfg, in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                     dilation=dilation, groups=groups, bias=bias)
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        if self.with_norm:
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_normalization_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)

        if self.with_act:
            if order.index('act') > order.index('conv'):
                act_channels = out_channels
            else:
                act_channels = in_channels
            self.act_name, act = build_activation_layer(act_cfg, act_channels)
            self.add_module(self.act_name, act)

        if self.with_dropout:
            self.dropout = nn.Dropout2d(p=dropout)

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    @property
    def activate(self):
        return getattr(self, self.act_name)

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_act:
                x = self.activate(x)
        if self.with_dropout:
            x = self.dropout(x)
        return x


class ConvModules(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias='auto',
                 conv_cfg=None, norm_cfg=None, act_cfg=None, order=('conv', 'norm', 'act'), dropouts=None, num_convs=1):
        super().__init__()

        if conv_cfg is None:
            conv_cfg = dict(type='Conv')
        if act_cfg is None:
            act_cfg = dict(type='Relu', inplace=True)
        if dropouts is not None:
            assert num_convs == len(dropouts)
            dropout = dropouts[0]
        else:
            dropout = None

        layers = [
            ConvModule(in_channels, out_channels, kernel_size, stride, padding,
                       dilation, groups, bias, conv_cfg, norm_cfg, act_cfg,
                       order, dropout),
        ]
        for ii in range(1, num_convs):
            if dropouts is not None:
                dropout = dropouts[ii]
            else:
                dropout = None
            layers.append(
                ConvModule(out_channels, out_channels, kernel_size, stride,
                           padding, dilation, groups, bias, conv_cfg, norm_cfg,
                           act_cfg,
                           order, dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        feat = self.block(x)
        return feat
