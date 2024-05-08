_base_ = [
    './upernet_augreg_adapter_large_512_160k_ade20k.py'
]
model = dict(decode_head=dict(type='UPerHead_ssa'))

evaluation = dict(interval=8000, metric='mIoU', save_best='mIoU')
data = dict(samples_per_gpu=2)