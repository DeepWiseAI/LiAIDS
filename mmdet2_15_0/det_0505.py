fp16 = dict(loss_scale=512.)
# model settings
model = dict(
    type='FasterRCNN',
    pretrained=None,
    backbone=dict(
        type='CSWin',
        embed_dim=64,
        img_size = 224,
        in_chans=9,
        #num_classes=1,
        depth=[1, 2, 9, 1],
        num_heads=[2, 4, 8, 16],
        split_size=[1,2,7,7],
        drop_path_rate = 0.2,
        mlp_ratio=4.
        ),
    neck=dict(
        type='FPN',
        in_channels=[64,128,256,512],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[1,3],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
        
# model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            #mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.0,
            nms=dict(type='nms', iou_threshold=0.1),
            max_per_img=100)))
# dataset settings
dataset_type = 'LiverDataset'
data_root = ''
# test_ct_root = '/data1/dongqi/shaoyifu/raw_data/INHOUSE/'
test_ct_root = ''
num_slice = 9
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=False, lung_input=True, num_slice=num_slice, zflip=True),
    dict(type='LoadAnnotations', with_bbox=True, skip_img_without_anno=False),
    dict(type='Resize',
         multiscale_mode='value',
         img_scale=[(384,384), (448,448), (512, 512), (576, 576), (640, 640)],
         keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),     
    dict(type='Rotate', level=10, max_rotate_angle=180, prob=0.5),   
    dict(type='Normalize', **img_norm_cfg, is_3d_input=True, num_slice=num_slice),  
    dict(type='Pad', size_divisor=32),  
    dict(type='DefaultFormatBundle', is_3d_input=True),       
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=False, lung_input=True, num_slice=num_slice, zflip=False),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(512, 512),
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg, is_3d_input=True, num_slice=num_slice),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img'], is_3d_input=True),
            dict(type='Collect', keys=['img']),
        ])
]
test_ct_pipeline = [
    dict(type='LoadImageFromTensor', to_float32=False, num_slice=num_slice),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg, is_3d_input=True, num_slice=num_slice),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img'], is_3d_input=True),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,  
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/',
        img_prefix=data_root + 'train_images/',
        pipeline=train_pipeline,
        filter_empty_gt=False), # Allow empty gt images    
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/',
        img_prefix=data_root + 'valid_images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/',
        img_prefix=data_root + 'valid_images/',
        pipeline=test_pipeline),
    test_ct=dict(
        type='LiverNiiCTDataset',
        sub_dirs='dummy',
        # image_root=test_ct_root + 'image_niis/',
        image_root=test_ct_root,
        sub_dir_list=test_ct_root,
        num_slice=num_slice,
        num_workers=20,    
        pipeline=test_ct_pipeline),
    test_ct_all=dict(
        type='LiverctDataset',
        sub_dir='dummy',
        # image_root=test_ct_root + 'image_niis/',
        image_root=test_ct_root,
        sub_dir_list=test_ct_root + 'test_ct_txts/outhouse_quzhou.txt',
        slice_expand=1,
        mask_root=None,
        ct_type='nii',
        pipeline=test_ct_pipeline))
evaluation = dict(interval=1, metric='bbox')
# optimizer
optimizer = dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)

optimizer_config = dict(grad_clip=None)
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
#custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
