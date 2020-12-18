import torch

from models import build_model
from .common import Common


class InferenceRunner(Common):
    def __init__(self, num_classes, out, cudnn_deterministic, cudnn_benchmark, cfg):
        cfg = cfg.copy()
        super().__init__(num_classes, cudnn_deterministic, cudnn_benchmark, out)

        self.multi_label = False
        self.transform = self.transform(cfg['transforms'])

        self.model = self._build_model(cfg['model'])
        self.model.eval()

    def _build_model(self, cfg):
        self.logger.info('Build model')
        model = build_model(cfg)
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
            model.cuda()
        return model

    def compute(self, output):
        if self.multi_label:
            output = output.sigmoid()
            output = torch.where(output >= 0.5,
                                 torch.full_like(output, 1),
                                 torch.full_like(output, 0)).long()

        else:
            output = output.softmax(dim=1)
            _, output = torch.max(output, dim=1)
        return output

    def __call__(self, image, masks):
        with torch.no_grad():
            image = self.transform(image=image, masks=masks)['image']
            image = image.unsqueeze(0)

            if self.use_gpu:
                image = image.cuda()

            output = self.model(image)
            output = self.compute(output)

            output = output.squeeze().cpu().numpy()

        return output
