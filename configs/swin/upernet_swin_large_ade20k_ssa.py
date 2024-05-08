_base_ = ['./upernet_swin_large_ade20k.py']

model = dict(decode_head=dict(type='UPerHead_ssa'))