_base_ = ['upernet_swin_tiny_4x4_80k_pc.py']

model = dict(decode_head=dict(type='UPerHead_ssa'))