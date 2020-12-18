import os
import random
import logging
import sys
import torch
import time

import numpy as np
from torch.backends import cudnn
import metrics.metrics as metrics

from torch.utils.data import DataLoader
from dataset.voc import VOCDataset
from transforms import build_transform
from utils.sampler import DefaultSampler
from utils.checkpoint import load_checkpoint


class Common:
    def __init__(self, num_classes, cudnn_deterministic, cudnn_benchmark, out):
        self.out = out
        self.use_gpu = self.device('0,1')
        self.num_classes = num_classes
        self.logger = logger(out=self.out)
        self.cudnn(cudnn_deterministic, cudnn_benchmark)
        self.seed(0)
        self.metric = self._build_metric()

    def device(self, gpu_id):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        self.gpu_num = torch.cuda.device_count()
        return True if torch.cuda.is_available() else False

    def seed(self, seed):
        if seed:
            self.logger.info('Set seed {}'.format(seed))
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    def cudnn(self, deterministic, benchmark):
        cudnn.deterministic = deterministic
        cudnn.benchmark = benchmark

    def dataloader(self, cfg):
        transform = build_transform(cfg['transforms'])
        dataset = cfg['dataset']
        dataset = VOCDataset(root=dataset['root'], imglist_name=dataset['imglist_name'], transform=transform)
        shuffle = cfg['dataloader'].pop('shuffle', False)
        sampler = DefaultSampler(dataset=dataset, shuffle=shuffle)
        return build_dataloader(self.gpu_num,
                                cfg['dataloader'],
                                dict(dataset=dataset,
                                     sampler=sampler))

    def _build_metric(self):
        mets = [metrics.IoU(num_classes=self.num_classes), metrics.MeanIoU(num_classes=self.num_classes)]
        return metrics.Compose(mets)

    def transform(self, config):
        return build_transform(config)

    def load_from_checkpoint(self, filename, map_location='default', strict=True):
        self.logger.info('Load checkpoint from {}'.format(filename))
        if map_location == 'default':
            if self.use_gpu:
                device_id = torch.cuda.current_device()
                map_location = lambda storage, loc: storage.cuda(device_id)
            else:
                map_location = 'cpu'
        return load_checkpoint(self.model, filename, map_location, strict)


def logger(out):
    format_ = '%(asctime)s - %(message)s'
    formatter = logging.Formatter(format_)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    rank = 0
    if rank == 0:
        pass

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    handlers = (
        dict(type='StreamHandler'),
        dict(type='FileHandler'),
    )
    for handler in handlers:
        if handler['type'] == 'FileHandler':
            fp = os.path.join(out, '%s.log' % timestamp)
            instance = logging.FileHandler(fp, 'w')
        else:
            instance = logging.StreamHandler(sys.stdout)
        instance.setFormatter(formatter)
        instance.setLevel('INFO')
        logger.addHandler(instance)
    return logger


def build_dataloader(num_gpus, cfg, default_args=None):
    args = cfg.copy()
    samples_per_gpu = args.pop('samples_per_gpu')
    workers_per_gpu = args.pop('workers_per_gpu')
    batch_size = num_gpus * samples_per_gpu
    num_workers = num_gpus * workers_per_gpu
    args.update({'batch_size': batch_size,
                 'num_workers': num_workers})
    args.pop('type')
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return DataLoader(**args)
