import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

from runners.train_runner import TrainRunner
import cv2

nclasses = 21
ignore_label = 255
img_norm_cfg = dict(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255.0)
test_size_h, test_size_w = 513, 513

norm_cfg = dict(type='BN')
multi_label = False
crop_size_h, crop_size_w = 513, 513
image_pad_value = (123.675, 116.280, 103.530)

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
        encoder=dict(
            backbone=dict(
                type='ResNet',
                arch='resnet101',
                pretrain=True,
                norm_cfg=norm_cfg,
            ),
        ),
        decoder=dict(
            type='GFPN',
            neck=[
                dict(
                    type='JunctionBlock',
                    fusion_method='concat',
                    top_down=dict(
                        from_layer='c5',
                        upsample=dict(
                            type='Upsample',
                            scale_factor=2,
                            scale_bias=-1,
                            mode='bilinear',
                            align_corners=True,
                        ),
                    ),
                    lateral=dict(from_layer='c4'),
                    post=dict(
                        type='ConvModules',
                        in_channels=3072,
                        out_channels=256,
                        kernel_size=3,
                        padding=1,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='Relu', inplace=True),
                        num_convs=2,
                    ),
                    to_layer='p4',
                ),  # 16
                dict(
                    type='JunctionBlock',
                    fusion_method='concat',
                    top_down=dict(
                        from_layer='p4',
                        upsample=dict(
                            type='Upsample',
                            scale_factor=2,
                            scale_bias=-1,
                            mode='bilinear',
                            align_corners=True,
                        ),
                    ),
                    lateral=dict(from_layer='c3'),
                    post=dict(
                        type='ConvModules',
                        in_channels=768,  # 256 + 512
                        out_channels=128,
                        kernel_size=3,
                        padding=1,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='Relu', inplace=True),
                        num_convs=2,
                    ),
                    to_layer='p3',
                ),
                dict(
                    type='JunctionBlock',
                    fusion_method='concat',
                    top_down=dict(
                        from_layer='p3',
                        upsample=dict(
                            type='Upsample',
                            scale_factor=2,
                            scale_bias=-1,
                            mode='bilinear',
                            align_corners=True,
                        ),
                    ),
                    lateral=dict(from_layer='c2'),
                    post=dict(
                        type='ConvModules',
                        in_channels=384,
                        out_channels=64,
                        kernel_size=3,
                        padding=1,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='Relu', inplace=True),
                        num_convs=2,
                    ),
                    to_layer='p2',
                ),
                dict(
                    type='JunctionBlock',
                    fusion_method='concat',
                    top_down=dict(
                        from_layer='p2',
                        upsample=dict(
                            type='Upsample',
                            scale_factor=2,
                            scale_bias=-1,
                            mode='bilinear',
                            align_corners=True,
                        ),
                    ),
                    lateral=dict(from_layer='c1'),
                    post=dict(
                        type='ConvModules',
                        in_channels=128,  # 64 + 64 same as resnet18
                        out_channels=32,
                        kernel_size=3,
                        padding=1,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='Relu', inplace=True),
                        num_convs=2,
                    ),
                    to_layer='p1',
                ),  # 2
                dict(
                    type='JunctionBlock',
                    top_down=dict(
                        from_layer='p1',
                        upsample=dict(
                            type='Upsample',
                            scale_factor=2,
                            scale_bias=-1,
                            mode='bilinear',
                            align_corners=True,
                        ),
                    ),
                    lateral=None,
                    post=dict(
                        type='ConvModules',
                        in_channels=32,
                        out_channels=16,
                        kernel_size=3,
                        padding=1,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='Relu', inplace=True),
                        num_convs=2,
                    ),
                    to_layer='p0',
                ),  # 1
            ]),
        head=dict(
            type='Head',
            in_channels=16,
            out_channels=nclasses,
            norm_cfg=norm_cfg,
            num_convs=0,
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
                dict(type='RandomScale', scale_limit=(0.5, 2),
                     interpolation=cv2.INTER_LINEAR),
                dict(type='PadIfNeeded', min_height=crop_size_h, min_width=crop_size_w,
                     value=image_pad_value, mask_value=ignore_label),
                dict(type='RandomCrop', height=crop_size_h, width=crop_size_w),
                dict(type='Rotate', limit=10, interpolation=cv2.INTER_LINEAR,
                     border_mode=cv2.BORDER_CONSTANT,
                     value=image_pad_value, mask_value=ignore_label, p=0.5),
                dict(type='GaussianBlur', blur_limit=7, p=0.5),
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
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    lr_scheduler=dict(type='PolyLR', max_epochs=max_epochs),
    max_epochs=max_epochs,
    trainval_ratio=1,
    log_interval=10,
    snapshot_interval=1,
    save_best=True,
)


def main():
    out = os.path.join(root_out, 'unet')
    os.makedirs(out, exist_ok=True)
    runner = TrainRunner(nclasses, out, False, True, train, inference)
    runner()


if __name__ == '__main__':
    main()
