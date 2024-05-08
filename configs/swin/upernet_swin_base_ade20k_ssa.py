_base_ = ['upernet_swin_base_ade20k.py']

model = dict(decode_head=dict(type='UPerHead_ssa'))


data = dict(samples_per_gpu=2)