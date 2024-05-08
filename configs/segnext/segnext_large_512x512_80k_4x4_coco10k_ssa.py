_base_ = [
    './segnext_large_512x512_80k_4x4_coco10k.py',
]
model = dict(
    decode_head=dict(type='LightHamHead_ssa')
)