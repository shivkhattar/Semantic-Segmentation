import torch
from torch.utils.data import Sampler


class DefaultSampler(Sampler):

    def __init__(self, dataset, shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            return iter(torch.randperm(len(self.dataset)).tolist())
        else:
            return iter(range(len(self.dataset)))

    def __len__(self):
        return len(self.dataset)
