_base_ = [
    './seaformer_base_480x480_80k_4x4_pc.py'
]
model = dict(
    decode_head=dict(
        type='LightHead_ssa',),)