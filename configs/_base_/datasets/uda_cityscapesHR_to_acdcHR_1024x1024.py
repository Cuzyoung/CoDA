
# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (1024, 1024)
cityscapes_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
acdc_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(1920, 1080)),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1920, 1080),
        # MultiScaleFlipAug is disabled by not providing img_ratios and
        # setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

"""
step1: 1200 easy_rfs(ref 1200 imgs) + 4000 twilight (ref 400 imgs)
step2: 1200 medium_hard(sd+adverse 2400 imgs) + 4000 night (400 imgs) + 1200 medium_hard + 400 night
step3: 48000 all_target --→ medium_hard + twilight + night (1200 + 1200 + 800 imgs)
"""


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='UDADataset',
        source=dict(
            type='CityscapesDataset',
            data_root='/share/home/dq070/hy-tmp/datasets/cityscapes/',
            img_dir='leftImg8bit/train',
            ann_dir='gtFine/train',
            pipeline=cityscapes_train_pipeline),
        target=dict(
            type='ACDCDataset',
            data_root='/share/home/dq070/hy-tmp/AS_id_all/all-condition',
            img_dir='sd_rfs',  #sd_rfs_twilight
            ann_dir='gt/train',
            pipeline=acdc_train_pipeline),
        night=dict(
            type='ACDCDataset',
            data_root='/share/home/dq070/hy-tmp/AS_id_all/night/night-img',
            img_dir='img/train',
            ann_dir='gt/train',
            pipeline=acdc_train_pipeline),
        all_target=dict(
            type='ACDCDataset',
            data_root='/share/home/dq070/hy-tmp/AS_id_all/all-condition/all-img',
            img_dir='sd_all',  # sd_all
            ann_dir='gt/train',
            pipeline=acdc_train_pipeline)),
    val=dict(
        type='ACDCDataset',
        data_root='/share/home/dq070/hy-tmp/AS_id_all/all-condition/all-img',
            img_dir='img/val',
            ann_dir='gt/val/labelTrainIds',
        pipeline=test_pipeline),
    test=dict(
        type='ACDCDataset',
        data_root='/share/home/dq070/hy-tmp/AS_id_all/all-condition/all-img',
            img_dir='img/test',
            ann_dir='gt/test/labelTrainIds',
        pipeline=test_pipeline))
