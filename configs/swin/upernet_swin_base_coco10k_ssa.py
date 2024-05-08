_base_ = ['upernet_swin_base_coco10k.py']

model = dict(decode_head=dict(type='UPerHead_ssa'))