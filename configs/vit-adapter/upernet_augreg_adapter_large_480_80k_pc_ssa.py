_base_ = [
    './upernet_augreg_adapter_large_480_80k_pc.py'
]
model = dict(decode_head=dict(type='UPerHead_ssa'))

evaluation = dict(interval=4000, metric='mIoU', save_best='mIoU')
data = dict(samples_per_gpu=4)