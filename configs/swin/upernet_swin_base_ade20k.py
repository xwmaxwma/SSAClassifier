_base_ = [
    './upernet_swin_tiny_ade20k.py'
]
checkpoint_file = 'ckpts/swin_base_patch4_window7_224_20220317-e9b98025.pth'  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32]),
    decode_head=dict(in_channels=[128, 256, 512, 1024], num_classes=150),
    auxiliary_head=dict(in_channels=512, num_classes=150))

data = dict(samples_per_gpu=4)
evaluation = dict(interval=8000, metric='mIoU', pre_eval=True)
