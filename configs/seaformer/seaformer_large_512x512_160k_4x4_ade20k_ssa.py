_base_ = [
    './seaformer_large_512x512_160k_4x4_ade20k.py'
]
model = dict(
    decode_head=dict(
        type='LightHead_ssa',),)