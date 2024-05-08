_base_ = ['afformer_base_512x512_80k_4x4_coco10k.py']

model = dict(
    decode_head=dict(
        type='CLS_ssa',),)