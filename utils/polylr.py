import warnings
import weakref
from functools import wraps

from torch.optim import Optimizer


class LRScheduler(object):
    _iter_based = True

    def __init__(self, optimizer, niter_per_epoch, last_iter=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        self.niter_per_epoch = niter_per_epoch
        if last_iter == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
            last_iter = 0
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified in "
                                   "param_groups[{}] when resuming an "
                                   "optimizer".format(i))
        self.base_lrs = list(
            map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_epoch = int(last_iter / niter_per_epoch)
        self.last_iter = None

        def with_counter(method):
            if getattr(method, '_with_counter', False):
                return method
            instance_ref = weakref.ref(method.__self__)
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self.optimizer._step_count = 0
        self._step_count = 0
        self.step(last_iter)

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if
                key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def step(self, iter_=None):
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn(
                    "Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                    "initialization. Please, make sure to call `optimizer.step()` before "
                    "`lr_scheduler.step()`. See more details at "
                    "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate",
                    UserWarning)

            elif self.optimizer._step_count < 1:
                warnings.warn(
                    "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                    "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                    "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                    "will save_result in PyTorch skipping the first value of the learning rate schedule."
                    "See more details at "
                    "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate",
                    UserWarning)
        self._step_count += 1

        if iter_ is None:
            iter_ = self.last_iter + 1
        self.last_iter = iter_
        self.last_epoch = int(iter_ / self.niter_per_epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class PolyLR(LRScheduler):
    def __init__(self, optimizer, niter_per_epoch, max_epochs, power=0.9,
                 last_iter=-1, warm_up=0):
        self.max_iters = niter_per_epoch * max_epochs
        self.power = power
        self.warm_up = warm_up
        super().__init__(optimizer, niter_per_epoch, last_iter)

    def get_lr(self):
        if self.last_iter < self.warm_up:
            multiplier = (self.last_iter / float(self.warm_up)) ** self.power
        else:
            multiplier = (1 - self.last_iter / float(
                self.max_iters)) ** self.power
        return [base_lr * multiplier for base_lr in self.base_lrs]
