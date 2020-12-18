import os
import time

import torch

from collections import OrderedDict


def weights_to_cpu(state_dict):
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    return state_dict_cpu


def save_checkpoint(model, filename, optimizer=None, lr_scheduler=None,
                    meta=None):
    if meta is None:
        meta = {}
    meta.update(time=time.asctime())

    file_dir = os.path.dirname(filename)
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)

    if hasattr(model, 'module'):
        model = model.module

    checkpoint = {
        'meta': meta,
        'state_dict': weights_to_cpu(model.state_dict())
    }
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    if lr_scheduler is not None:
        checkpoint['lr_scheduler'] = lr_scheduler.state_dict()
    torch.save(checkpoint, filename)


def load_checkpoint(model, filename, map_location=None, strict=False):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location=map_location)
        if isinstance(checkpoint, OrderedDict):
            state_dict = checkpoint
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            raise RuntimeError(
                'No state_dict found in checkpoint file {}'.format(filename))
        if hasattr(model, 'module'):
            model.module.load_state_dict(state_dict, strict=strict)
        else:
            model.load_state_dict(state_dict, strict=strict)
        return checkpoint
    else:
        raise RuntimeError(
            'No checkpoint file found in path {}'.format(filename))
