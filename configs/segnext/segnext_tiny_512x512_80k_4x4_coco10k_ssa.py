_base_ = ['segnext_tiny_512x512_80k_4x4_coco10k.py']


model = dict(
    decode_head=dict(type='LightHamHead_ssa')
)