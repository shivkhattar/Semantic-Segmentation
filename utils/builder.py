from .conv_module import ConvModule, ConvModules
from .upsample import Upsample


def build_module(cfg):
    args = cfg.copy()
    if args['type'] == 'ConvModule':
        args.pop('type')
        return ConvModule(**args)
    elif args['type'] == 'ConvModules':
        args.pop('type')
        return ConvModules(**args)
    elif args['type'] == 'Upsample':
        args.pop('type')
        return Upsample(**args)
