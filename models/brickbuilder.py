from models.bricks import BlockJunction, BlockCollect, BlockFusion
import torch.nn as nn


def brick(config):
    args = config.copy()
    type = args.pop('type')
    if type == 'JunctionBlock':
        return BlockJunction(**args)
    elif type == 'FusionBlock':
        return BlockFusion(**args)
    elif type == 'CollectBlock':
        return BlockCollect(**args)


def bricks(config):
    bricks = nn.ModuleList()
    for brick_config in config:
        bricks.append(brick(brick_config))
    return bricks
