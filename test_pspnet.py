import os
import sys
from runners.test_runner import TestRunner
import train_pspnet as base
from test.testutil import parse_args

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

test = dict(
    data=dict(
        dataset=dict(
            type=base.dataset_type,
            root=base.dataset_root,
            imglist_name='val.txt',
            multi_label=base.multi_label,
        ),
        transforms=base.inference['transforms'],
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
    out = os.path.join(base.root_out, 'pspnet')
    os.makedirs(out, exist_ok=True)
    runner = TestRunner(base.nclasses, out, False, True, test, base.inference)
    runner.load_from_checkpoint(args.checkpoint)
    runner()


if __name__ == '__main__':
    main()
