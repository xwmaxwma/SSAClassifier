_base_ = [
    'upernet_swin_tiny_4x4_80k_pc.py'
]
checkpoint_file = 'ckpts/swin_large_patch4_window7_224_22k_20220412-aeecf2aa.pth'  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        pretrain_img_size=224,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7),
    decode_head=dict(in_channels=[192, 384, 768, 1536]),
    auxiliary_head=dict(in_channels=768))
