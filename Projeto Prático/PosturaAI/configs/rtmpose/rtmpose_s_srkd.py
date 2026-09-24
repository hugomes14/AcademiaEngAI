"""Standalone MMPose 1.x RTMPose-S config for the 30-point SRKD dataset.

Architecture and SimCC settings follow OpenMMLab's RTMPose-S COCO 256x192
config. Stage-specific freezing, learning rates, and epoch caps are applied by
``src.posturaai.training.build_stage_config`` before MMEngine constructs a run.
The dataset's zero sigmas are parser sentinels; no OKS or CocoMetric is used.
"""

default_scope = "mmpose"
custom_imports = dict(imports=["src.posturaai.training"], allow_failed_imports=False)

# MMPose uses width, height order. The model sees 192 x 256 pixels.
input_size = (192, 256)
codec = dict(
    type="SimCCLabel",
    input_size=input_size,
    sigma=(4.9, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False,
)

official_backbone_url = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
    "cspnext-s_udp-aic-coco_210e-256x192-92f5a029_20230130.pth"
)
model = dict(
    type="TopdownPoseEstimator",
    data_preprocessor=dict(
        type="PoseDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        _scope_="mmdet",
        type="CSPNeXt",
        arch="P5",
        expand_ratio=0.5,
        deepen_factor=0.33,
        widen_factor=0.5,
        out_indices=(4,),
        channel_attention=True,
        norm_cfg=dict(type="BN"),
        act_cfg=dict(type="SiLU"),
        frozen_stages=4,
        init_cfg=dict(type="Pretrained", prefix="backbone.", checkpoint=official_backbone_url),
    ),
    head=dict(
        type="RTMCCHead",
        in_channels=512,
        out_channels=30,
        input_size=input_size,
        in_featuremap_size=(6, 8),
        simcc_split_ratio=2.0,
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.0,
            drop_path=0.0,
            act_fn="SiLU",
            use_rel_bias=False,
            pos_enc=False,
        ),
        loss=dict(type="KLDiscretLoss", use_target_weight=True, beta=10.0, label_softmax=True),
        decoder=codec,
    ),
    test_cfg=dict(flip_test=True),
)

dataset_type = "CocoDataset"
data_mode = "topdown"
data_root = "."
dataset_metainfo = dict(from_file="configs/datasets/srkd.py")

train_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="RandomFlip", direction="horizontal"),
    dict(type="RandomBBoxTransform", scale_factor=[0.85, 1.15], rotate_factor=25),
    dict(type="TopdownAffine", input_size=input_size),
    dict(type="mmdet.YOLOXHSVRandomAug"),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]
stage2_train_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="RandomFlip", direction="horizontal"),
    dict(type="RandomBBoxTransform", scale_factor=[0.85, 1.15], rotate_factor=25),
    dict(type="TopdownAffine", input_size=input_size),
    dict(type="mmdet.YOLOXHSVRandomAug"),
    dict(
        type="Albumentation",
        transforms=[
            dict(type="Blur", blur_limit=3, p=0.10),
            dict(type="ImageCompression", quality_range=(85, 100), p=0.10),
            dict(
                type="CoarseDropout",
                num_holes_range=(1, 1),
                hole_height_range=(0.05, 0.15),
                hole_width_range=(0.05, 0.15),
                p=0.10,
            ),
        ],
    ),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]
val_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="TopdownAffine", input_size=input_size),
    dict(type="PackPoseInputs"),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file="data/srkd/derived/train.json",
        data_prefix=dict(img=""),
        metainfo=dataset_metainfo,
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file="data/srkd/derived/val.json",
        data_prefix=dict(img=""),
        metainfo=dataset_metainfo,
        test_mode=True,
        pipeline=val_pipeline,
    ),
)

val_evaluator = dict(type="SRKDBboxHeightMetric", prefix="srkd")
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=10, val_interval=1)
val_cfg = dict(type="ValLoop")
optim_wrapper = dict(
    type="OptimWrapper",
    accumulative_counts=4,
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
)
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-3, by_epoch=False, begin=0, end=500),
    dict(type="CosineAnnealingLR", eta_min=1e-6, begin=1, end=10, by_epoch=True),
]
randomness = dict(seed=42, deterministic=True)
auto_scale_lr = dict(enable=False, base_batch_size=64)

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        by_epoch=True,
        save_best="srkd/NME",
        rule="less",
        save_last=True,
        max_keep_ckpts=2,
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
)
custom_hooks = [
    dict(type="EarlyStoppingHook", monitor="srkd/NME", rule="less", min_delta=0.0, patience=4, strict=True),
]
env_cfg = dict(cudnn_benchmark=False, mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0))
log_level = "INFO"
log_processor = dict(by_epoch=True)
