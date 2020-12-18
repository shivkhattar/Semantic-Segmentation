import os
import sys
from test.testutil import parse_args
from runners.test_runner import TestRunner
import train_unet as unet

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

test = dict(
    data=dict(
        dataset=dict(
            type=unet.dataset_type,
            root=unet.dataset_root,
            imglist_name='val.txt',
            multi_label=False,
        ),
        transforms=unet.inference['transforms'],
        sampler=dict(
            type='DefaultSampler',
        ),
        dataloader=dict(
            type='DataLoader',
            samples_per_gpu=4,
            workers_per_gpu=4,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        ),
    ),
)


def main():
    args = parse_args()
    out = os.path.join(unet.root_out, 'unet')
    os.makedirs(out, exist_ok=True)
    runner = TestRunner(unet.nclasses, out, False, True, test, unet.inference)
    runner.load_from_checkpoint(args.checkpoint)
    runner()


if __name__ == '__main__':
    main()
