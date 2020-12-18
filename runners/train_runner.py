import os
from collections import OrderedDict
from collections.abc import Iterable

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optims
from utils.checkpoint import save_checkpoint
from utils.polylr import PolyLR
from .inference_runner import InferenceRunner


class TrainRunner(InferenceRunner):
    def __init__(self, num_classes, out, cudnn_deterministic, cudnn_benchmark, train_cfg, inference_cfg):
        super().__init__(num_classes, out, cudnn_deterministic, cudnn_benchmark, inference_cfg)
        self.train_dataloader = self.dataloader(train_cfg['data']['train'])
        self.val_dataloader = self.dataloader(train_cfg['data']['val']) if 'val' in train_cfg['data'] else None
        opt_cfg = train_cfg['optimizer']
        self.optimizer = optims.SGD(lr=opt_cfg['lr'], momentum=opt_cfg['momentum'],
                                    weight_decay=opt_cfg['weight_decay'],
                                    params=self.model.parameters())
        criterion_cfg = train_cfg['criterion']
        self.criterion = nn.CrossEntropyLoss(ignore_index=criterion_cfg['ignore_index'])
        self.lr_scheduler = self._build_lr_scheduler(train_cfg['lr_scheduler'])
        self.max_epochs = train_cfg['max_epochs']
        self.log_interval = train_cfg.get('log_interval', 10)
        self.trainval_ratio = train_cfg.get('trainval_ratio', -1)
        self.snapshot_interval = train_cfg.get('snapshot_interval', -1)
        self.save_best = train_cfg.get('save_best', True)
        self.iter_based = hasattr(self.lr_scheduler, '_iter_based')

        assert self.out is not None
        assert self.log_interval > 0

        self.best = OrderedDict()
        self.iter = 0

        if train_cfg.get('resume'):
            self.resume(**train_cfg['resume'])

    def _build_optimizer(self, cfg):
        return optims.SGD(lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'],
                          params=self.model.parameters())

    def _build_lr_scheduler(self, cfg):
        return PolyLR(max_epochs=cfg['max_epochs'], optimizer=self.optimizer, niter_per_epoch=len(
            self.train_dataloader))

    def train(self):
        self.metric.reset()
        self.model.train()

        self.logger.info('Epoch {}'.format(self.epoch + 1))
        for idx, (image, mask) in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            if self.use_gpu:
                image = image.cuda()
                mask = mask.cuda()

            output = self.model(image)
            loss = self.criterion(output, mask)

            loss.backward()
            self.optimizer.step()
            self.iter += 1

            with torch.no_grad():
                output = self.compute(output)
                loss_found = loss.item()

                self.metric(output.cpu().numpy(), mask.cpu().numpy())
                res = self.metric.get_metrics()

            if self.iter % self.log_interval == 0:
                self.logger.info(
                    'Training: Epoch {}, Iter {}, Loss {:.4f}, {}'.format(
                        self.epoch + 1, self.iter, loss_found, ', '.join(
                            ['{}: {}'.format(k, np.round(v, 4)) for k, v in
                             res.items()])))

            if self.iter_based:
                self.lr_scheduler.step()

        if not self.iter_based:
            self.lr_scheduler.step()

    def validate(self):
        self.metric.reset()
        self.model.eval()

        res = {}

        self.logger.info('Starting to validate')
        with torch.no_grad():
            for idx, (image, mask) in enumerate(self.val_dataloader):
                if self.use_gpu:
                    image = image.cuda()
                    mask = mask.cuda()

                output = self.model(image)
                output = self.compute(output)

                self.metric(output.cpu().numpy(), mask.cpu().numpy())
                res = self.metric.get_metrics()

                if (idx + 1) % self.log_interval == 0:
                    self.logger.info('Validation: Iter {}, {}'.format(
                        idx + 1,
                        ', '.join(
                            ['{}: {}'.format(k, np.round(v, 4)) for k, v in
                             res.items()])))

        return res

    def __call__(self):
        for _ in range(self.epoch, self.max_epochs):
            if hasattr(self.train_dataloader.sampler, 'set_epoch'):
                self.train_dataloader.sampler.set_epoch(self.epoch)

            self.train()

            if self.trainval_ratio > 0 and \
                    self.epoch % self.trainval_ratio == 0 and \
                    self.val_dataloader:
                res = self.validate()
                for k, v in res.items():
                    if isinstance(v, (int, float)):
                        if k not in self.best:
                            self.best[k] = 0.0
                        if self.best[k] <= res[k]:
                            self.best[k] = res[k]
                            if self.save_best:
                                self.save_checkpoint(
                                    self.out, 'best_{}.pth'.format(k),
                                    meta=dict(best=self.best))
                self.logger.info(', '.join(
                    ['Best {}: {}'.format(k, v) for k, v in self.best.items()]))

            if self.snapshot_interval > 0 and \
                    self.epoch % self.snapshot_interval == 0:
                self.logger.info('Snapshot')
                self.save_checkpoint(
                    self.out, 'epoch_{}.pth'.format(self.epoch),
                    meta=dict(best=self.best))

    @property
    def epoch(self):
        return self.lr_scheduler.last_epoch

    @epoch.setter
    def epoch(self, val):
        self.lr_scheduler.last_epoch = val

    @property
    def lr(self):
        lr = [x['lr'] for x in self.optimizer.param_groups]
        return np.array(lr)

    @lr.setter
    def lr(self, val):
        for idx, param in enumerate(self.optimizer.param_groups):
            if isinstance(val, Iterable):
                param['lr'] = val[idx]
            else:
                param['lr'] = val

    def save_checkpoint(self, directory, filename, meta=None):
        filepath = os.path.join(directory, filename)
        optimizer = self.optimizer
        lr_scheduler = self.lr_scheduler

        self.logger.info('Saving checkpoint {}'.format(filename))
        if meta is None:
            meta = dict(epoch=self.epoch, iter=self.iter, lr=self.lr)
        else:
            meta.update(epoch=self.epoch, iter=self.iter, lr=self.lr)
        save_checkpoint(self.model, filepath, optimizer, lr_scheduler, meta)

    def resume(self, checkpoint, resume_optimizer=False,
               resume_lr_scheduler=False,
               map_location='default'):
        checkpoint = self.load_from_checkpoint(checkpoint,
                                               map_location=map_location)
        if resume_lr_scheduler and 'lr_scheduler' in checkpoint:
            self.logger.info('Resume lr scheduler')
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        if resume_optimizer and 'optimizer' in checkpoint:
            self.logger.info('Resume optimizer')
            self.optimizer.load_state_dict(checkpoint['optimizer'])
