import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

from runners.train_runner import TrainRunner
import cv2

nclasses = 21
ignore_label = 255
image_pad_value = (123.675, 116.280, 103.530)
crop_size_h, crop_size_w = 513, 513
test_size_h, test_size_w = 513, 513
img_norm_cfg = dict(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255.0)
norm_cfg = dict(type='BN')
multi_label = False

inference = dict(
    gpu_id='0,1',
    multi_label=multi_label,
    transforms=[
        dict(type='PadIfNeeded', min_height=test_size_h, min_width=test_size_w,
             value=image_pad_value, mask_value=ignore_label),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='ToTensor'),
    ],
    model=dict(
        # model/encoder
        encoder=dict(
            backbone=dict(
                type='ResNet',
                arch='resnet101',
                replace_stride_with_dilation=[False, False, True],
                multi_grid=[1, 2, 4],
                norm_cfg=norm_cfg,
            ),
            enhance=dict(
                type='ASPP',
                from_layer='c5',
                to_layer='enhance',
                in_channels=2048,
                out_channels=256,
                atrous_rates=[6, 12, 18],
                mode='bilinear',
                align_corners=True,
                norm_cfg=norm_cfg,
                dropout=0.1,
            ),
        ),
        collect=dict(type='CollectBlock', from_layer='enhance'),
        # model/head
        head=dict(
            type='Head',
            in_channels=256,
            inter_channels=256,
            out_channels=nclasses,
            norm_cfg=norm_cfg,
            num_convs=1,
            upsample=dict(
                type='Upsample',
                scale_factor=16,
                scale_bias=-15,
                mode='bilinear',
                align_corners=True
            ),
        )
    )
)

root_out = 'out'
dataset_type = 'VOCDataset'
dataset_root = '../vedaseg/data/VOCdevkit/VOC2012/'

max_epochs = 50

train = dict(
    data=dict(
        train=dict(
            dataset=dict(
                type=dataset_type,
                root=dataset_root,
                imglist_name='trainaug.txt',
                multi_label=multi_label,
            ),
            transforms=[
                dict(type='RandomScale', scale_limit=(0.5, 2), scale_step=0.25,
                     interpolation=cv2.INTER_LINEAR),
                dict(type='PadIfNeeded', min_height=crop_size_h, min_width=crop_size_w,
                     value=image_pad_value, mask_value=ignore_label),
                dict(type='RandomCrop', height=crop_size_h, width=crop_size_w),
                dict(type='HorizontalFlip', p=0.5),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='ToTensor'),
            ],
            sampler=dict(
                type='DefaultSampler',
            ),
            dataloader=dict(
                type='DataLoader',
                samples_per_gpu=8,
                workers_per_gpu=4,
                shuffle=True,
                drop_last=True,
                pin_memory=True,
            ),
        ),
        val=dict(
            dataset=dict(
                type=dataset_type,
                root=dataset_root,
                imglist_name='val.txt',
                multi_label=multi_label,
            ),
            transforms=inference['transforms'],
            sampler=dict(
                type='DefaultSampler',
            ),
            dataloader=dict(
                type='DataLoader',
                samples_per_gpu=8,
                workers_per_gpu=4,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
            ),
        ),
    ),
    resume=None,
    criterion=dict(type='CrossEntropyLoss', ignore_index=ignore_label),
    optimizer=dict(type='SGD', lr=0.007, momentum=0.9, weight_decay=0.0001),
    lr_scheduler=dict(type='PolyLR', max_epochs=max_epochs),
    max_epochs=max_epochs,
    trainval_ratio=1,
    log_interval=10,
    snapshot_interval=5,
    save_best=True,
)


def main():
    out = os.path.join(root_out, 'deeplab')
    os.makedirs(out, exist_ok=True)
    runner = TrainRunner(nclasses, out, False, True, train, inference)
    runner()


if __name__ == '__main__':
    main()
