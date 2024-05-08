_base_ = ['upernet_swin_large_4x4_80k_coco10k.py']

model = dict(decode_head=dict(type='UPerHead_ssa'))