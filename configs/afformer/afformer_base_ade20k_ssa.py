_base_ = ['afformer_base_ade20k.py']

model = dict(
    decode_head=dict(
        type='CLS_ssa',),)