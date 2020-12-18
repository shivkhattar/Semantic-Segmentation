from models.resnet import ResNet
from models.aspp import ASPP
from models.ppm import PPM
import torch.nn as nn
from models.gfpn import GFPN
from models.head import Head
from .brickbuilder import brick


def build_model(config):
    encoder = build_encoder(config.get('encoder'))
    if config.get('decoder'):
        mid_layer = build_decoder(config.get('decoder'))
        assert 'collect' not in config
    else:
        assert 'collect' in config
        mid_layer = brick(config.get('collect'))
    head = build_head(config['head'])
    model = nn.Sequential(encoder, mid_layer, head)
    return model


def build_encoder(cfg):
    backbone = build_backbone(cfg['backbone'])
    enhance_cfg = cfg.get('enhance')
    if enhance_cfg:
        enhance_module = build_enhance(enhance_cfg)
        encoder = nn.Sequential(backbone, enhance_module)
    else:
        encoder = backbone
    return encoder


def build_backbone(backbonecfg):
    args = backbonecfg.copy()
    type = args.pop('type')
    if type == 'ResNet':
        return ResNet(**args)


def build_enhance(enhancecfg):
    args = enhancecfg.copy()
    type = args.pop('type')
    if type == 'ASPP':
        return ASPP(**args)
    elif type == 'PPM':
        return PPM(**args)


def build_decoder(decodercfg):
    args = decodercfg.copy()
    type = args.pop('type')
    if type == 'GFPN':
        return GFPN(**args)


def build_head(cfg):
    args = cfg.copy()
    args.pop('type')
    return Head(**args)
