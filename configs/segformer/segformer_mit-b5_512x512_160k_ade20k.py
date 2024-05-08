_base_ = ['./segformer_mit-b0_512x512_160k_ade20k.py']

checkpoint = 'ckpts/mit_b5_20220624-658746d9.pth'  # noqa

# model settings
model = dict(
    pretrained=checkpoint,
    backbone=dict(
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[3, 6, 40, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]))

data = dict(
    samples_per_gpu=4, 
    workers_per_gpu=4)

evaluation = dict(interval=8000, metric='mIoU')
checkpoint_config = dict(by_epoch=False, interval=8000)
