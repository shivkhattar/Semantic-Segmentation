import numpy as np
from abc import ABCMeta, abstractmethod


class MetricBase(object, metaclass=ABCMeta):

    def __init__(self):
        super().__init__()
        self.reset()

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def compute(self, pred, target):
        pass

    @abstractmethod
    def update(self, n=1):
        pass

    @abstractmethod
    def get_metrics(self):
        pass

    @staticmethod
    def validate_type(pred, target):
        assert type(pred) == np.ndarray and type(target) == np.ndarray

    def check(self, pred, target):
        self.validate_type(pred, target)
        self.validate_match(pred, target)

    @staticmethod
    def validate_match(pred, target):
        assert pred.shape[0] == target.shape[0] and pred.shape[-2:-1] == target.shape[-2:-1]

    def __call__(self, pred, target):
        self.check(pred, target)
        current_state = self.compute(pred, target)
        self.update()
        return current_state


class ConfusionMatrix(MetricBase):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        super().__init__()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes,) * 2)

    def compute(self, pred, target):
        mask = (target >= 0) & (target < self.num_classes)
        self.current_state = np.bincount(
            self.num_classes * target[mask].astype('int') + pred[mask],
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        return self.current_state

    def update(self, n=1):
        self.confusion_matrix += self.current_state

    def get_metrics(self):
        accumulate_state = {
            'confusion matrix': self.confusion_matrix
        }
        return accumulate_state


class IoU(ConfusionMatrix):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        super().__init__(num_classes=self.num_classes)

    def get_metrics(self):
        iou = self.confusion_matrix.diagonal() / (
                self.confusion_matrix.sum(axis=0) + self.confusion_matrix.sum(axis=1) -
                self.confusion_matrix.diagonal() + np.finfo(np.float32).eps)
        return {
            'IoUs': iou
        }


class MeanIoU(IoU):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        super().__init__(num_classes=self.num_classes)

    def get_metrics(self):
        ious = (super().get_metrics())['IoUs']
        miou = np.nanmean(ious)
        accumulate_state = {
            'mIoU': miou
        }
        return accumulate_state


class Accuracy(ConfusionMatrix):
    def __init__(self, num_classes, average='pixel'):
        self.num_classes = num_classes
        self.average = average
        super().__init__(num_classes=self.num_classes)

    def get_metrics(self):
        if self.average == 'pixel':
            accuracy = self.confusion_matrix.diagonal().sum() / (
                    self.confusion_matrix.sum() + 1e-15)
        elif self.average == 'class':
            accuracy_class = self.confusion_matrix.diagonal() / self.confusion_matrix.sum(axis=1)
            accuracy = np.nanmean(accuracy_class)

        accumulate_state = {
            'accuracy': accuracy
        }
        return accumulate_state


class DiceScore(ConfusionMatrix):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        super().__init__(self.num_classes)

    def get_metrics(self):
        dice_score = 2.0 * self.confusion_matrix.diagonal() / (self.confusion_matrix.sum(axis=1) +
                                                               self.confusion_matrix.sum(axis=0) +
                                                               np.finfo(np.float32).eps)
        accumulate_state = {
            'dice_score': dice_score
        }
        return accumulate_state


class Compose:
    def __init__(self, metrics):
        self.metrics = metrics

    def reset(self):
        for m in self.metrics:
            m.reset()

    def get_metrics(self):
        res = dict()
        for m in self.metrics:
            results = m.get_metrics()
            res.update(results)
        return res

    def __call__(self, pred, target):
        for m in self.metrics:
            m(pred, target)
