_base_ = ['upernet_swin_tiny_4x4_80k_coco10k.py']

model = dict(decode_head=dict(type='UPerHead_ssa'))