import torch
import numpy as np
import metrics.metrics as metrics

from .inference_runner import InferenceRunner


class TestRunner(InferenceRunner):
    def __init__(self, num_classes, out, cudnn_deterministic, cudnn_benchmark, test_config, inference_cfg):
        super().__init__(num_classes, out, cudnn_deterministic, cudnn_benchmark, inference_cfg)
        self.test_dataloader = self.dataloader(test_config['data'])
        self.metric = self.test_metrics()

    def __call__(self):
        self.metric.reset()
        self.model.eval()
        res = {}
        self.logger.info('Starting to Test')
        with torch.no_grad():
            for idx, (image, mask) in enumerate(self.test_dataloader):
                if self.use_gpu:
                    image = image.cuda()
                    mask = mask.cuda()

                output = self.model(image)
                output = self.compute(output)
                self.metric(output.cpu().numpy(), mask.cpu().numpy())
                res = self.metric.get_metrics()
                self.logger.info('Testing: Iter {}, {}'.format(
                    idx + 1,
                    ', '.join(['{}: {}'.format(k, np.round(v, 4)) for k, v in
                               res.items()])))
        self.logger.info('Test Result: {}'.format(', '.join(
            ['{}: {}'.format(k, np.round(v, 4)) for k, v in res.items()])))

        return res

    def test_metrics(self):
        mets = [metrics.IoU(num_classes=self.num_classes), metrics.MeanIoU(num_classes=self.num_classes),
                metrics.Accuracy(num_classes=self.num_classes), metrics.DiceScore(num_classes=self.num_classes)]
        return metrics.Compose(mets)
